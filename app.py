from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# -----------------------------------------------------------------------------
# Load models once at startup
# -----------------------------------------------------------------------------
PRICE_MODEL_PATH = MODELS_DIR / "price_model.joblib"
DEMAND_MODEL_PATH = MODELS_DIR / "court_demand_model.joblib"
MATCHMAKING_MODEL_PATH = MODELS_DIR / "matchmaking_model.joblib"
PLAYERS_CSV = DATA_DIR / "players.csv"
PAIRS_CSV = DATA_DIR / "pairs.csv"
USER_ID_MAP_CSV = DATA_DIR / "user_uuid_map.csv"

price_bundle: Dict[str, Any] = joblib.load(PRICE_MODEL_PATH)
price_model = price_bundle["model"]
PRICE_FEATURE_COLS: List[str] = price_bundle["feature_cols"]
PRICE_INTERVAL_HALF_WIDTH: float = float(price_bundle.get("interval_half_width", 0))
PRICE_METADATA: Dict[str, Any] = price_bundle.get("metadata", {})
PRICE_LABELING_RULE: str = price_bundle.get(
    "labeling_rule",
    "asking < lower => Underpriced; asking > upper => Overpriced; else Fair",
)

demand_bundle: Dict[str, Any] = joblib.load(DEMAND_MODEL_PATH)
demand_pipeline = demand_bundle["pipeline"]
DEMAND_FEATURES: List[str] = demand_bundle["features"]
DEMAND_CLASSES = demand_bundle.get("classes", [])

matchmaking_bundle: Dict[str, Any] = joblib.load(MATCHMAKING_MODEL_PATH)
matchmaking_model = matchmaking_bundle["model"]
MATCHMAKING_FEATURE_COLS: List[str] = matchmaking_bundle["feature_cols"]
MATCHMAKING_LAMBDA_PROBABILITY = float(matchmaking_bundle["lambda_probability"])
MATCHMAKING_METRICS = matchmaking_bundle.get("metrics", {})

players = pd.read_csv(PLAYERS_CSV)
pairs = pd.read_csv(PAIRS_CSV)
players["user_id"] = players["user_id"].astype(int)
pairs["user_id"] = pairs["user_id"].astype(int)
pairs["candidate_id"] = pairs["candidate_id"].astype(int)
player_by_id = players.set_index("user_id")
pair_lookup = (
    pairs.drop_duplicates(["user_id", "candidate_id"], keep="first")
    .set_index(["user_id", "candidate_id"])
    .sort_index()
)

USER_ID_NAMESPACE = uuid5(NAMESPACE_URL, "tennisfinder.matchmaking.user-id")
INTERNAL_TO_PUBLIC_USER_ID: Dict[int, UUID] = {}
PUBLIC_TO_INTERNAL_USER_ID: Dict[UUID, int] = {}
for raw_user_id in players["user_id"].tolist():
    internal_user_id = int(raw_user_id)
    public_user_id = uuid5(USER_ID_NAMESPACE, str(internal_user_id))
    INTERNAL_TO_PUBLIC_USER_ID[internal_user_id] = public_user_id
    PUBLIC_TO_INTERNAL_USER_ID[public_user_id] = internal_user_id


def _save_user_id_mappings() -> None:
    rows = [
        {"internal_user_id": internal_user_id, "public_user_id": str(public_user_id)}
        for internal_user_id, public_user_id in sorted(INTERNAL_TO_PUBLIC_USER_ID.items())
    ]
    pd.DataFrame(rows).to_csv(USER_ID_MAP_CSV, index=False)


def _apply_user_id_mapping(
    internal_user_id: int,
    public_user_id: UUID,
    internal_to_public: Dict[int, UUID],
    public_to_internal: Dict[UUID, int],
    *,
    allow_reassign: bool,
) -> None:
    if internal_user_id not in player_by_id.index:
        raise HTTPException(status_code=400, detail=f"Unknown internal_user_id: {internal_user_id}")

    existing_public_user_id = internal_to_public.get(internal_user_id)
    default_public_user_id = uuid5(USER_ID_NAMESPACE, str(internal_user_id))
    replacing_default_generated_mapping = (
        existing_public_user_id is not None and existing_public_user_id == default_public_user_id
    )
    if (
        existing_public_user_id is not None
        and existing_public_user_id != public_user_id
        and not allow_reassign
        and not replacing_default_generated_mapping
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"internal_user_id {internal_user_id} is already mapped to {existing_public_user_id}. "
                "Set allow_reassign=true to replace it."
            ),
        )

    mapped_internal_user_id = public_to_internal.get(public_user_id)
    if mapped_internal_user_id is not None and mapped_internal_user_id != internal_user_id:
        raise HTTPException(
            status_code=409,
            detail=f"public_user_id {public_user_id} is already mapped to internal_user_id {mapped_internal_user_id}",
        )

    if existing_public_user_id is not None and existing_public_user_id != public_user_id:
        public_to_internal.pop(existing_public_user_id, None)

    internal_to_public[internal_user_id] = public_user_id
    public_to_internal[public_user_id] = internal_user_id


if USER_ID_MAP_CSV.exists():
    user_id_map_df = pd.read_csv(USER_ID_MAP_CSV)
    required_columns = {"internal_user_id", "public_user_id"}
    if not required_columns.issubset(set(user_id_map_df.columns)):
        raise ValueError(f"{USER_ID_MAP_CSV} must contain columns: {sorted(required_columns)}")
    for record in user_id_map_df.to_dict(orient="records"):
        _apply_user_id_mapping(
            internal_user_id=int(record["internal_user_id"]),
            public_user_id=UUID(str(record["public_user_id"])),
            internal_to_public=INTERNAL_TO_PUBLIC_USER_ID,
            public_to_internal=PUBLIC_TO_INTERNAL_USER_ID,
            allow_reassign=True,
        )

app = FastAPI(
    title="TennisFinder Combined ML API",
    version="1.0.0",
    description=(
        "One FastAPI backend for all TennisFinder ML models: marketplace price, "
        "court demand, and smart matchmaking."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Shared/root endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "TennisFinder Combined ML API",
        "docs": "/docs",
        "health": "/health",
        "models": {
            "price": {
                "metadata": "/price/metadata",
                "predict": "/price/predict",
                "predict_batch": "/price/predict-batch",
            },
            "court_demand": {
                "metadata": "/court-demand/metadata",
                "predict": "/court-demand/predict",
                "predict_batch": "/court-demand/predict-batch",
            },
            "matchmaking": {
                "metrics": "/matchmaking/metrics",
                "id_mapping_status": "/matchmaking/id-mapping/status",
                "id_mapping_list": "/matchmaking/id-mapping",
                "id_mapping_sync": "/matchmaking/id-mapping/sync",
                "predict_pair": "/matchmaking/predict-pair",
                "recommend": "/matchmaking/recommend/{user_id}",
                "recommend_open_match": "/matchmaking/recommend-open-match",
            },
        },
    }


@app.get("/health")
def combined_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": {
            "price": PRICE_MODEL_PATH.exists(),
            "court_demand": DEMAND_MODEL_PATH.exists(),
            "matchmaking": MATCHMAKING_MODEL_PATH.exists(),
        },
        "matchmaking_players_loaded": int(len(players)),
        "matchmaking_pairs_loaded": int(len(pairs)),
    }

# -----------------------------------------------------------------------------
# Price model
# -----------------------------------------------------------------------------
class PriceItemFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(..., examples=["Racket"])
    condition: str = Field(..., examples=["Used - Excellent"])
    brand: str = Field(..., examples=["Wilson"])
    model: str = Field(..., examples=["Pro Staff 97"])
    flaw: str = Field(..., examples=["None"])
    age_months: float = Field(..., ge=0, examples=[12])
    original_price: float = Field(..., gt=0, examples=[14500])
    asking_price: Optional[float] = Field(
        None,
        ge=0,
        description="Optional seller asking price. If supplied, returns Underpriced/Fair/Overpriced.",
    )

    @field_validator("category", "condition", "brand", "model", "flaw")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text fields cannot be empty")
        return value


class PriceBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[PriceItemFeatures] = Field(..., min_length=1, max_length=200)


def _price_label(asking_price: Optional[float], lower: float, upper: float) -> Optional[str]:
    if asking_price is None:
        return None
    if asking_price < lower:
        return "Underpriced"
    if asking_price > upper:
        return "Overpriced"
    return "Fair"


def _predict_price_one(item: PriceItemFeatures) -> Dict[str, Any]:
    row = item.model_dump()
    asking_price = row.pop("asking_price", None)
    df = pd.DataFrame([{col: row[col] for col in PRICE_FEATURE_COLS}])
    predicted = float(price_model.predict(df)[0])
    lower = max(0.0, predicted - PRICE_INTERVAL_HALF_WIDTH)
    upper = predicted + PRICE_INTERVAL_HALF_WIDTH
    return {
        "recommended_price": round(predicted, 2),
        "price_range": {"lower": round(lower, 2), "upper": round(upper, 2)},
        "currency": "EGP",
        "asking_price": asking_price,
        "label": _price_label(asking_price, lower, upper),
    }


@app.get("/price/health")
def price_health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": PRICE_MODEL_PATH.exists(), "feature_cols": PRICE_FEATURE_COLS}


@app.get("/price/metadata")
def price_metadata() -> Dict[str, Any]:
    return {
        "feature_cols": PRICE_FEATURE_COLS,
        "interval_half_width": PRICE_INTERVAL_HALF_WIDTH,
        "labeling_rule": PRICE_LABELING_RULE,
        "training_rows": price_bundle.get("training_rows"),
        "calibration_rows": price_bundle.get("calibration_rows"),
        "metadata": PRICE_METADATA,
    }


@app.post("/price/predict")
def price_predict(item: PriceItemFeatures) -> Dict[str, Any]:
    try:
        return _predict_price_one(item)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Price prediction failed: {exc}") from exc


@app.post("/price/predict-batch")
def price_predict_batch(request: PriceBatchRequest) -> Dict[str, Any]:
    try:
        return {"results": [_predict_price_one(item) for item in request.items]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Price batch prediction failed: {exc}") from exc

# -----------------------------------------------------------------------------
# Court demand model
# -----------------------------------------------------------------------------
class DemandPredictionRequest(BaseModel):
    court_id: UUID = Field(..., examples=["de1c19bb-b98f-4c57-b684-370ac0283dd0"])
    court_type: str = Field(..., examples=["Hard"])
    location: str = Field(..., examples=["Giza"])
    hour: int = Field(..., ge=0, le=23, examples=[9])
    price: float = Field(..., ge=0, examples=[157])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])
    month: int = Field(..., ge=1, le=12, examples=[2])


class DemandPredictionResponse(BaseModel):
    predicted_demand_level: str
    probabilities: Optional[dict] = None


class BatchDemandPredictionRequest(BaseModel):
    items: List[DemandPredictionRequest] = Field(..., min_length=1, max_length=200)


def _predict_demand_one(payload: DemandPredictionRequest) -> DemandPredictionResponse:
    row = payload.model_dump(mode="json")
    df = pd.DataFrame([row], columns=DEMAND_FEATURES)
    try:
        prediction = demand_pipeline.predict(df)[0]
        probabilities = None
        if hasattr(demand_pipeline, "predict_proba"):
            proba = demand_pipeline.predict_proba(df)[0]
            model_classes = demand_pipeline.named_steps["model"].classes_
            probabilities = {str(label): float(value) for label, value in zip(model_classes, proba)}
        return DemandPredictionResponse(
            predicted_demand_level=str(prediction),
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Demand prediction failed: {exc}") from exc


@app.get("/court-demand/health")
def demand_health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": DEMAND_MODEL_PATH.exists()}


@app.get("/court-demand/metadata")
def demand_metadata() -> Dict[str, Any]:
    return {
        "target": demand_bundle.get("target"),
        "features": DEMAND_FEATURES,
        "classes": DEMAND_CLASSES,
        "sample_input": demand_bundle.get("sample_input"),
        "metrics": {
            "accuracy": demand_bundle.get("metrics", {}).get("accuracy"),
            "f1_weighted": demand_bundle.get("metrics", {}).get("f1_weighted"),
        },
    }


@app.post("/court-demand/predict", response_model=DemandPredictionResponse)
def demand_predict(payload: DemandPredictionRequest):
    return _predict_demand_one(payload)


@app.post("/court-demand/predict-batch")
def demand_predict_batch(payload: BatchDemandPredictionRequest):
    return {"predictions": [_predict_demand_one(item).model_dump() for item in payload.items]}

# -----------------------------------------------------------------------------
# Smart matchmaking model
# -----------------------------------------------------------------------------
class PairPredictionRequest(BaseModel):
    user_id: UUID = Field(..., example="cd8de5f2-a8f4-4928-aa84-9e4ce71a607b")
    candidate_id: UUID = Field(..., example="6db67299-e7c7-4c92-a84d-3478a34ccb88")
    common_opponents: Optional[int] = Field(None, ge=0, example=2)
    prev_encounters: Optional[int] = Field(None, ge=0, example=1)


class MatchCandidate(BaseModel):
    candidate_id: UUID = Field(..., example="6db67299-e7c7-4c92-a84d-3478a34ccb88")
    common_opponents: Optional[int] = Field(None, ge=0, example=2)
    prev_encounters: Optional[int] = Field(None, ge=0, example=1)


class OpenMatchRecommendationRequest(BaseModel):
    match_id: Optional[UUID] = Field(None, example="7cb3ccaf-cc96-4270-8aa0-c71665f8c652")
    user_id: UUID = Field(
        ...,
        description="The user who created the open match",
        example="cd8de5f2-a8f4-4928-aa84-9e4ce71a607b",
    )
    candidates: List[MatchCandidate] = Field(
        ...,
        description="Only candidates who sent a request to join the open match",
        min_length=1,
        example=[
            {"candidate_id": "6db67299-e7c7-4c92-a84d-3478a34ccb88"},
            {"candidate_id": "f4021d30-eac0-4db5-b5d8-ab95db04af72"},
            {"candidate_id": "27ad95fa-f726-45dc-ac3d-d51fd2f2dcc0"},
        ],
    )
    sort_by: Literal["final_score", "compatibility_probability"] = "final_score"


class CustomPairFeatures(BaseModel):
    skill_gap: float
    distance_km: float
    time_overlap: int = Field(..., ge=0, le=1)
    style_match: int = Field(..., ge=0, le=1)
    mutual_readiness: float = Field(..., ge=0, le=1)
    mutual_reliability: float = Field(..., ge=0, le=1)
    recency_bonus: float = Field(..., ge=0, le=1)
    common_opponents: int = Field(0, ge=0)
    prev_encounters: int = Field(0, ge=0)


class UserIdMappingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    internal_user_id: int = Field(..., ge=0, example=12)
    public_user_id: UUID = Field(..., example="cd8de5f2-a8f4-4928-aa84-9e4ce71a607b")


class UserIdMappingSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mappings: List[UserIdMappingItem] = Field(..., min_length=1, max_length=5000)
    allow_reassign: bool = Field(
        False,
        description="When false, any remap is rejected. When true, existing mapping for that internal id can be replaced.",
    )
    persist: bool = Field(
        True,
        description="When true, writes final mapping to data/user_uuid_map.csv",
    )
    dry_run: bool = Field(
        False,
        description="When true, validates everything but does not mutate in-memory or persisted mappings.",
    )


class PublicUserIdResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    public_user_ids: List[UUID] = Field(..., min_length=1, max_length=5000)


def _haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2 * radius_km * np.arcsin(np.sqrt(a)))


def _public_user_id(internal_user_id: int) -> UUID:
    if internal_user_id not in INTERNAL_TO_PUBLIC_USER_ID:
        raise HTTPException(status_code=404, detail=f"Player {internal_user_id} was not found")
    return INTERNAL_TO_PUBLIC_USER_ID[internal_user_id]


def _is_default_generated_mapping(internal_user_id: int, public_user_id: UUID) -> bool:
    return public_user_id == uuid5(USER_ID_NAMESPACE, str(internal_user_id))


def _resolve_internal_user_id_for_public(
    public_user_id: UUID,
    taken_internal_user_ids: set[int],
    assignable_internal_user_ids: List[int],
) -> Optional[int]:
    if not assignable_internal_user_ids:
        return None

    start_index = public_user_id.int % len(assignable_internal_user_ids)
    for step in range(len(assignable_internal_user_ids)):
        idx = (start_index + step) % len(assignable_internal_user_ids)
        internal_user_id = assignable_internal_user_ids[idx]
        if internal_user_id in taken_internal_user_ids:
            continue
        return internal_user_id
    return None


def _internal_user_id(user_id: UUID) -> int:
    if user_id not in PUBLIC_TO_INTERNAL_USER_ID:
        raise HTTPException(status_code=404, detail=f"Player {user_id} was not found")
    return PUBLIC_TO_INTERNAL_USER_ID[user_id]


def _get_player(user_id: UUID):
    internal_user_id = _internal_user_id(user_id)
    if internal_user_id not in player_by_id.index:
        raise HTTPException(status_code=404, detail=f"Player {user_id} was not found")
    return player_by_id.loc[internal_user_id]


def _build_pair_features(user_id: UUID, candidate_id: UUID, common_opponents=None, prev_encounters=None):
    source = _get_player(user_id)
    candidate = _get_player(candidate_id)
    internal_user_id = _internal_user_id(user_id)
    internal_candidate_id = _internal_user_id(candidate_id)

    skill_gap = abs(float(source["skill_level"]) - float(candidate["skill_level"]))
    distance_km = _haversine(source["lat"], source["lng"], candidate["lat"], candidate["lng"])
    time_overlap = int(source["preferred_slot"] == candidate["preferred_slot"])
    style_match = int(source["play_style"] == candidate["play_style"])
    mutual_readiness = float(np.sqrt(source["play_readiness_score"] * candidate["play_readiness_score"]))
    mutual_reliability = float(np.sqrt(source["reliability_score"] * candidate["reliability_score"]))
    recency_bonus = float(1 - min(source["last_active_days"], candidate["last_active_days"]) / 15.0)

    try:
        known_pair = pair_lookup.loc[(internal_user_id, internal_candidate_id)]
        default_common = int(known_pair["common_opponents"])
        default_prev = int(known_pair["prev_encounters"])
    except KeyError:
        default_common = 0
        default_prev = 0

    return {
        "skill_gap": skill_gap,
        "distance_km": distance_km,
        "time_overlap": time_overlap,
        "style_match": style_match,
        "mutual_readiness": mutual_readiness,
        "mutual_reliability": mutual_reliability,
        "recency_bonus": recency_bonus,
        "common_opponents": int(default_common if common_opponents is None else common_opponents),
        "prev_encounters": int(default_prev if prev_encounters is None else prev_encounters),
    }


def _score_matchmaking_rows(rows: List[dict]):
    X = pd.DataFrame(rows)[MATCHMAKING_FEATURE_COLS]
    probabilities = matchmaking_model.predict_proba(X)[:, 1]
    final_scores = MATCHMAKING_LAMBDA_PROBABILITY * probabilities + (1 - MATCHMAKING_LAMBDA_PROBABILITY) * X[
        "mutual_readiness"
    ].to_numpy()
    return probabilities, final_scores


def _recommendation_label(score: float) -> str:
    if score >= 0.80:
        return "Highly Recommended"
    if score >= 0.50:
        return "Recommended"
    return "Not Recommended"


def _recommendation_decision(label: str) -> bool:
    return label in {"Recommended", "Highly Recommended"}


@app.get("/matchmaking/health")
def matchmaking_health():
    return {
        "status": "ok",
        "model_loaded": MATCHMAKING_MODEL_PATH.exists(),
        "players_loaded": int(len(players)),
        "pairs_loaded": int(len(pairs)),
    }


@app.get("/matchmaking/metrics")
def matchmaking_metrics():
    return MATCHMAKING_METRICS


@app.get("/matchmaking/id-mapping/status")
def matchmaking_id_mapping_status():
    return {
        "status": "ok",
        "total_mapped_players": len(INTERNAL_TO_PUBLIC_USER_ID),
        "mapping_file": str(USER_ID_MAP_CSV),
        "mapping_file_exists": USER_ID_MAP_CSV.exists(),
    }


@app.get("/matchmaking/id-mapping")
def matchmaking_list_id_mappings(limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    all_internal_ids = sorted(INTERNAL_TO_PUBLIC_USER_ID.keys())
    selected_internal_ids = all_internal_ids[offset : offset + limit]
    rows = [
        {"internal_user_id": internal_user_id, "public_user_id": INTERNAL_TO_PUBLIC_USER_ID[internal_user_id]}
        for internal_user_id in selected_internal_ids
    ]
    return {"count": len(rows), "total": len(all_internal_ids), "mappings": rows}


@app.get("/matchmaking/id-mapping/by-internal/{internal_user_id}")
def matchmaking_id_mapping_by_internal(internal_user_id: int):
    if internal_user_id not in INTERNAL_TO_PUBLIC_USER_ID:
        raise HTTPException(status_code=404, detail=f"No mapping found for internal_user_id {internal_user_id}")
    return {
        "internal_user_id": internal_user_id,
        "public_user_id": INTERNAL_TO_PUBLIC_USER_ID[internal_user_id],
    }


@app.get("/matchmaking/id-mapping/by-public/{public_user_id}")
def matchmaking_id_mapping_by_public(public_user_id: UUID):
    if public_user_id not in PUBLIC_TO_INTERNAL_USER_ID:
        raise HTTPException(status_code=404, detail=f"No mapping found for public_user_id {public_user_id}")
    return {
        "internal_user_id": PUBLIC_TO_INTERNAL_USER_ID[public_user_id],
        "public_user_id": public_user_id,
    }


@app.post("/matchmaking/id-mapping/resolve")
def matchmaking_resolve_public_user_ids(payload: PublicUserIdResolveRequest):
    resolved = []
    unknown_public_user_ids = []
    taken_internal_user_ids: set[int] = set()
    assignable_internal_user_ids = [
        int(internal_user_id)
        for internal_user_id, mapped_public_user_id in INTERNAL_TO_PUBLIC_USER_ID.items()
        if _is_default_generated_mapping(int(internal_user_id), mapped_public_user_id)
    ]
    if not assignable_internal_user_ids:
        assignable_internal_user_ids = [int(internal_user_id) for internal_user_id in INTERNAL_TO_PUBLIC_USER_ID.keys()]

    for public_user_id in payload.public_user_ids:
        internal_user_id = PUBLIC_TO_INTERNAL_USER_ID.get(public_user_id)
        if internal_user_id is None:
            internal_user_id = _resolve_internal_user_id_for_public(
                public_user_id=public_user_id,
                taken_internal_user_ids=taken_internal_user_ids,
                assignable_internal_user_ids=assignable_internal_user_ids,
            )
            if internal_user_id is None:
                unknown_public_user_ids.append(public_user_id)
                continue

        taken_internal_user_ids.add(int(internal_user_id))
        resolved.append({"internal_user_id": internal_user_id, "public_user_id": public_user_id})
    return {
        "count": len(resolved),
        "total_requested": len(payload.public_user_ids),
        "resolved": resolved,
        "unknown_public_user_ids": unknown_public_user_ids,
    }


@app.post("/matchmaking/id-mapping/sync")
def matchmaking_sync_id_mappings(payload: UserIdMappingSyncRequest):
    next_internal_to_public = dict(INTERNAL_TO_PUBLIC_USER_ID)
    next_public_to_internal = dict(PUBLIC_TO_INTERNAL_USER_ID)

    for item in payload.mappings:
        _apply_user_id_mapping(
            internal_user_id=item.internal_user_id,
            public_user_id=item.public_user_id,
            internal_to_public=next_internal_to_public,
            public_to_internal=next_public_to_internal,
            allow_reassign=payload.allow_reassign,
        )

    if payload.dry_run:
        return {
            "status": "validated",
            "dry_run": True,
            "applied_count": len(payload.mappings),
            "persisted": False,
        }

    INTERNAL_TO_PUBLIC_USER_ID.clear()
    INTERNAL_TO_PUBLIC_USER_ID.update(next_internal_to_public)
    PUBLIC_TO_INTERNAL_USER_ID.clear()
    PUBLIC_TO_INTERNAL_USER_ID.update(next_public_to_internal)

    if payload.persist:
        _save_user_id_mappings()

    return {
        "status": "applied",
        "dry_run": False,
        "applied_count": len(payload.mappings),
        "persisted": payload.persist,
        "total_mapped_players": len(INTERNAL_TO_PUBLIC_USER_ID),
    }


@app.get("/matchmaking/players")
def matchmaking_list_players(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    data = players.iloc[offset : offset + limit].to_dict(orient="records")
    for row in data:
        row["user_id"] = _public_user_id(int(row["user_id"]))
    return {"count": len(data), "total": int(len(players)), "players": data}


@app.post("/matchmaking/players/sync")
def matchmaking_sync_players(payload: UserIdMappingSyncRequest):
    # Backward-compatible alias for backend clients using the previous contract path.
    return matchmaking_sync_id_mappings(payload)


@app.get("/matchmaking/players/{user_id}")
def matchmaking_read_player(user_id: UUID):
    player = _get_player(user_id)
    output = player.to_dict()
    output["user_id"] = user_id
    return output


@app.post("/matchmaking/predict-pair")
def matchmaking_predict_pair(req: PairPredictionRequest):
    if req.user_id == req.candidate_id:
        raise HTTPException(status_code=400, detail="user_id and candidate_id must be different")
    features = _build_pair_features(req.user_id, req.candidate_id, req.common_opponents, req.prev_encounters)
    probability, final_score = _score_matchmaking_rows([features])
    score = float(final_score[0])
    label = _recommendation_label(score)
    return {
        "user_id": req.user_id,
        "candidate_id": req.candidate_id,
        "compatibility_probability": round(float(probability[0]), 6),
        "final_score": round(score, 6),
        "recommendation_label": label,
        "is_recommended": _recommendation_decision(label),
        "features": features,
    }


@app.post("/matchmaking/predict-features")
def matchmaking_predict_features(features: CustomPairFeatures):
    row = features.model_dump()
    probability, final_score = _score_matchmaking_rows([row])
    score = float(final_score[0])
    label = _recommendation_label(score)
    return {
        "compatibility_probability": round(float(probability[0]), 6),
        "final_score": round(score, 6),
        "recommendation_label": label,
        "is_recommended": _recommendation_decision(label),
        "features": row,
    }


@app.get("/matchmaking/recommend/{user_id}")
def matchmaking_recommend(
    user_id: UUID,
    top_k: int = Query(10, ge=1, le=100),
    max_distance_km: Optional[float] = Query(None, gt=0),
):
    internal_user_id = _internal_user_id(user_id)
    _get_player(user_id)
    all_candidate_internal_ids = [int(uid) for uid in players["user_id"].tolist() if int(uid) != internal_user_id]
    backend_mapped_candidate_internal_ids = [
        candidate_internal_id
        for candidate_internal_id in all_candidate_internal_ids
        if INTERNAL_TO_PUBLIC_USER_ID.get(candidate_internal_id) is not None
        and INTERNAL_TO_PUBLIC_USER_ID[candidate_internal_id].version == 4
    ]
    candidate_internal_ids = (
        backend_mapped_candidate_internal_ids
        if backend_mapped_candidate_internal_ids
        else all_candidate_internal_ids
    )
    rows = []
    ids = []
    for candidate_internal_id in candidate_internal_ids:
        candidate_id = _public_user_id(candidate_internal_id)
        features = _build_pair_features(user_id, candidate_id)
        if max_distance_km is not None and features["distance_km"] > max_distance_km:
            continue
        rows.append(features)
        ids.append(candidate_id)

    if not rows:
        return {"user_id": user_id, "count": 0, "recommendations": []}

    probabilities, final_scores = _score_matchmaking_rows(rows)
    result = pd.DataFrame(rows)
    result["user_id"] = user_id
    result["candidate_id"] = ids
    result["compatibility_probability"] = probabilities
    result["final_score"] = final_scores
    result["recommendation_label"] = [_recommendation_label(float(score)) for score in final_scores]
    result["is_recommended"] = [_recommendation_decision(label) for label in result["recommendation_label"]]

    candidate_profiles = players[
        [
            "user_id",
            "skill_level",
            "preferred_slot",
            "play_style",
            "play_readiness_score",
            "reliability_score",
            "last_active_days",
        ]
    ]
    candidate_profiles["candidate_id"] = candidate_profiles["user_id"].map(
        lambda raw_user_id: _public_user_id(int(raw_user_id))
    )
    candidate_profiles = candidate_profiles.drop(columns=["user_id"])
    result = result.merge(candidate_profiles, on="candidate_id", how="left")
    result = result.sort_values("final_score", ascending=False).head(top_k)

    clean_records = []
    for record in result.to_dict(orient="records"):
        clean_records.append(
            {key: (round(float(value), 6) if isinstance(value, (float, np.floating)) else value) for key, value in record.items()}
        )
    return {"user_id": user_id, "count": len(clean_records), "recommendations": clean_records}


@app.post("/matchmaking/recommend-open-match")
def matchmaking_recommend_open_match(req: OpenMatchRecommendationRequest):
    _get_player(req.user_id)

    seen = set()
    rows = []
    candidate_ids = []

    for candidate in req.candidates:
        candidate_id = candidate.candidate_id
        if candidate_id == req.user_id:
            raise HTTPException(status_code=400, detail="The match creator cannot be scored as their own candidate")
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        _get_player(candidate_id)
        rows.append(_build_pair_features(req.user_id, candidate_id, candidate.common_opponents, candidate.prev_encounters))
        candidate_ids.append(candidate_id)

    probabilities, final_scores = _score_matchmaking_rows(rows)
    result = pd.DataFrame(rows)
    result["match_id"] = req.match_id
    result["user_id"] = req.user_id
    result["candidate_id"] = candidate_ids
    result["compatibility_probability"] = probabilities
    result["final_score"] = final_scores
    result["recommendation_label"] = [_recommendation_label(float(score)) for score in final_scores]
    result["is_recommended"] = [_recommendation_decision(label) for label in result["recommendation_label"]]

    candidate_profiles = players[
        [
            "user_id",
            "skill_level",
            "preferred_slot",
            "play_style",
            "play_readiness_score",
            "reliability_score",
            "last_active_days",
        ]
    ]
    candidate_profiles["candidate_id"] = candidate_profiles["user_id"].map(
        lambda raw_user_id: _public_user_id(int(raw_user_id))
    )
    candidate_profiles = candidate_profiles.drop(columns=["user_id"])
    result = result.merge(candidate_profiles, on="candidate_id", how="left")
    result = result.sort_values(req.sort_by, ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    clean_records = []
    for record in result.to_dict(orient="records"):
        clean_records.append(
            {key: (round(float(value), 6) if isinstance(value, (float, np.floating)) else value) for key, value in record.items()}
        )

    return {
        "match_id": req.match_id,
        "user_id": req.user_id,
        "compared_candidate_ids": candidate_ids,
        "count": len(clean_records),
        "label_thresholds": {
            "Not Recommended": "final_score < 0.50",
            "Recommended": "0.50 <= final_score < 0.80",
            "Highly Recommended": "final_score >= 0.80",
        },
        "recommendations": clean_records,
    }
