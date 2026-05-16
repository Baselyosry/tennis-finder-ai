from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import floor
from pathlib import Path
import re
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
PRICE_MODEL_PATH = MODELS_DIR / "price_model_catalog_augmented_2026.joblib"
DEMAND_MODEL_PATH = MODELS_DIR / "court_demand_model.joblib"
MATCHMAKING_MODEL_PATH = MODELS_DIR / "matchmaking_model.joblib"
PLAYERS_CSV = DATA_DIR / "players.csv"
PAIRS_CSV = DATA_DIR / "pairs.csv"
USER_ID_MAP_CSV = DATA_DIR / "user_uuid_map.csv"
TENNIS_PRODUCT_CATALOG_CSV = DATA_DIR / "tennis_product_catalog.csv"
MARKETPLACE_ATTR_FIELDS = ["category", "condition", "brand", "model", "flaw", "age_months"]
MARKETPLACE_BRAND_MODEL_FIELDS = ["brand", "model"]
MARKETPLACE_CATEGORY_BRAND_MODEL_FIELDS = ["category", "brand", "model"]
MARKETPLACE_CATEGORY_CONDITION_FIELDS = ["category", "condition"]
MARKETPLACE_CATEGORY_FIELDS = ["category"]
TENNIS_PRODUCT_CATALOG_FIELDS = [
    "category",
    "brand",
    "model",
    "original_price",
    "currency",
    "source_name",
    "source_url",
    "last_checked",
    "aliases",
]
TENNIS_PRODUCT_REFERENCE_FIELDS = [
    "category",
    "brand",
    "model_name",
    "production_year",
    "egypt_original_price_egp_est",
    "global_source_url",
    "egypt_source_url",
    "aliases_for_fuzzy_matching",
]

price_bundle: Dict[str, Any] = joblib.load(PRICE_MODEL_PATH)
price_model = price_bundle["model"]
PRICE_FEATURE_COLS: List[str] = price_bundle["feature_cols"]
PRICE_INTERVAL_HALF_WIDTH: float = float(price_bundle.get("interval_half_width", 0))
PRICE_METADATA: Dict[str, Any] = price_bundle.get("metadata", {})
PRICE_LABELING_RULE: str = price_bundle.get(
    "labeling_rule",
    "asking < lower => Underpriced; asking > upper => Overpriced; else Fair",
)


@dataclass(frozen=True)
class MarketplacePriceCatalog:
    original_price_by_attr: Dict[tuple[Any, ...], float]
    original_price_by_category_brand_model: Dict[tuple[Any, ...], float]
    original_price_by_brand_model: Dict[tuple[Any, ...], float]
    original_price_by_category_condition: Dict[tuple[Any, ...], float]
    original_price_by_category: Dict[tuple[Any, ...], float]
    canonical_brand_by_key: Dict[str, str]
    canonical_model_by_brand_model_key: Dict[tuple[str, str], str]
    model_names_by_brand_key: Dict[str, List[str]]
    global_original_price: float
    raw_rows: int


@dataclass(frozen=True)
class MarketplaceOriginalPriceMatch:
    original_price: float
    match_level: str
    canonical_brand: str
    canonical_model: str
    dataset: Optional[str]
    is_estimated: bool
    warnings: List[Dict[str, str]]
    catalog_year: Optional[int] = None
    canonical_category: Optional[str] = None
    canonical_condition: Optional[str] = None
    canonical_flaw: Optional[str] = None


@dataclass(frozen=True)
class CatalogPriceRecord:
    category: str
    brand: str
    model: str
    original_price: float
    production_year: Optional[int]


@dataclass(frozen=True)
class TennisProductCatalog:
    records_by_category_brand_model: Dict[tuple[Any, ...], List[CatalogPriceRecord]]
    records_by_brand_model: Dict[tuple[Any, ...], List[CatalogPriceRecord]]
    canonical_brand_by_key: Dict[str, str]
    canonical_model_by_brand_model_key: Dict[tuple[str, str], str]
    raw_rows: int
    schema_name: str


@dataclass(frozen=True)
class PriceFeatureRow:
    category: str
    condition: str
    brand: str
    model: str
    flaw: str
    age_months: float
    original_price: float


def _normalize_marketplace_text(value: Any, *, allow_blank: bool = False) -> str:
    text = str(value).strip()
    if not text and not allow_blank:
        raise ValueError("Marketplace attribute text fields cannot be empty")
    return text


def _marketplace_text_key(value: Any, *, allow_blank: bool = False) -> str:
    return _normalize_marketplace_text(value, allow_blank=allow_blank).casefold()


def _catalog_model_lookup_text(value: Any) -> str:
    text = _normalize_marketplace_text(value).casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bv\s*\d{1,2}\b", "", text)
    text = re.sub(r"\bversion\s*\d{1,2}\b", "", text)
    text = re.sub(r"\b(size|length)\s*\d{2}(\.\d+)?\s*(in|inch|inches|cm)?\b", "", text)
    text = re.sub(r"\b\d{2}(\.\d+)?\s*(in|inch|inches)\b", "", text)
    text = re.sub(r"\bgrip\s*(size)?\s*\d\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _catalog_model_candidates(value: Any) -> List[str]:
    raw = _normalize_marketplace_text(value)
    cleaned = _catalog_model_lookup_text(raw)
    candidates = [raw]
    if cleaned and cleaned != raw.casefold():
        candidates.append(cleaned)
    seen: set[str] = set()
    unique: List[str] = []
    for candidate in candidates:
        key = _marketplace_text_key(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _marketplace_match_key_from_attrs(
    attrs: Dict[str, Any],
    fields: List[str],
    *,
    allow_blank_text: bool = False,
) -> tuple[Any, ...]:
    parts: List[Any] = []
    for field in fields:
        if field == "age_months":
            parts.append(round(float(attrs[field]), 6))
        else:
            parts.append(_marketplace_text_key(attrs[field], allow_blank=allow_blank_text))
    return tuple(parts)


def _resolve_price_training_csv_path(bundle: Dict[str, Any]) -> Path:
    raw = bundle.get("dataset_path")
    if raw is not None and str(raw).strip():
        path = Path(str(raw).strip())
        if path.is_file():
            return path.resolve()
        name = path.name
        for base in (MODELS_DIR, DATA_DIR, BASE_DIR):
            candidate = base / name
            if candidate.is_file():
                return candidate.resolve()

    for candidate in DATA_DIR.glob("marketplace_price_dataset_egypt*.csv"):
        if candidate.is_file():
            return candidate.resolve()

    if raw is None or not str(raw).strip():
        raise ValueError(
            f"{PRICE_MODEL_PATH.name} has no dataset_path and no marketplace_price_dataset_egypt*.csv "
            f"file was found under {DATA_DIR}; cannot locate the marketplace training CSV."
        )
    raise FileNotFoundError(
        f"Marketplace training CSV not found. The model bundle references {str(raw)!r}. "
        f"Place a file named {name!r} next to the weights (for example under {MODELS_DIR} or {DATA_DIR})."
    )


def _marketplace_price_index(
    df: pd.DataFrame,
    fields: List[str],
    *,
    allow_blank_text: bool = False,
) -> Dict[tuple[Any, ...], float]:
    aggregated = df.groupby(fields, as_index=False, sort=False, dropna=False)["original_price"].median()
    index: Dict[tuple[Any, ...], float] = {}
    for record in aggregated.to_dict(orient="records"):
        key = _marketplace_match_key_from_attrs(record, fields, allow_blank_text=allow_blank_text)
        index[key] = float(record["original_price"])
    return index


def _price_warning(message: str) -> Dict[str, str]:
    return {"code": "ORIGINAL_PRICE_ESTIMATED", "message": message}


def _round_to_nearest_50(value: float) -> int:
    return int(floor((float(value) / 50.0) + 0.5) * 50)


PRICE_CAP_RATIOS = {
    ("New", "None"): 1.00,
    ("New", ""): 1.00,
    ("Like New", "None"): 0.95,
    ("Like New", ""): 0.95,
    ("Like New", "Minor"): 0.90,
    ("Used", "None"): 0.85,
    ("Used", ""): 0.85,
    ("Used", "Minor"): 0.78,
    ("Used", "Moderate"): 0.65,
    ("Used", "Major"): 0.50,
}


def _price_cap_ratio(condition: Any, flaw: Any) -> float:
    condition_text = _normalize_marketplace_text(condition, allow_blank=True) or "Used"
    flaw_text = _normalize_marketplace_text(flaw, allow_blank=True) or "None"

    exact = PRICE_CAP_RATIOS.get((condition_text, flaw_text))
    if exact is not None:
        return exact

    condition_default = PRICE_CAP_RATIOS.get((condition_text, "None"))
    if condition_default is not None:
        return condition_default

    return 0.80


def _max_recommended_price(feature_row: Dict[str, Any]) -> float:
    original_price = float(feature_row["original_price"])
    ratio = _price_cap_ratio(feature_row["condition"], feature_row["flaw"])
    return max(0.0, original_price * ratio)


def _round_to_nearest_50_not_above(value: float, max_value: float) -> int:
    rounded = _round_to_nearest_50(value)
    if rounded > max_value:
        return int(floor(float(max_value) / 50.0) * 50)
    return rounded


def _usd_to_egp_rate() -> float:
    base_metadata = PRICE_METADATA.get("base_model_metadata", {})
    return float(PRICE_METADATA.get("usd_to_egp_seed_rate", base_metadata.get("usd_to_egp_seed_rate", 53.1)))


def _price_metadata_categories() -> List[str]:
    base_metadata = PRICE_METADATA.get("base_model_metadata", {})
    return [str(value) for value in PRICE_METADATA.get("categories", base_metadata.get("categories", []))]


def _load_marketplace_original_prices_from_training_csv(csv_path: Path) -> MarketplacePriceCatalog:
    df = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    required = set(MARKETPLACE_ATTR_FIELDS + ["original_price"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    raw_rows = int(len(df))
    df = df.dropna(subset=["original_price"])
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
    df = df[df["original_price"] > 0]
    if df.empty:
        raise ValueError(f"{csv_path} does not contain any usable original_price values.")

    canonical_brand_by_key: Dict[str, str] = {}
    canonical_model_by_brand_model_key: Dict[tuple[str, str], str] = {}
    models_by_brand_key: Dict[str, set[str]] = {}
    for record in df[["brand", "model"]].drop_duplicates().to_dict(orient="records"):
        brand = _normalize_marketplace_text(record["brand"])
        model_name = _normalize_marketplace_text(record["model"])
        brand_key = _marketplace_text_key(brand)
        model_key = _marketplace_text_key(model_name)
        canonical_brand_by_key.setdefault(brand_key, brand)
        canonical_model_by_brand_model_key.setdefault((brand_key, model_key), model_name)
        models_by_brand_key.setdefault(brand_key, set()).add(model_name)

    return MarketplacePriceCatalog(
        original_price_by_attr=_marketplace_price_index(
            df,
            MARKETPLACE_ATTR_FIELDS,
            allow_blank_text=True,
        ),
        original_price_by_category_brand_model=_marketplace_price_index(
            df,
            MARKETPLACE_CATEGORY_BRAND_MODEL_FIELDS,
            allow_blank_text=True,
        ),
        original_price_by_brand_model=_marketplace_price_index(
            df,
            MARKETPLACE_BRAND_MODEL_FIELDS,
            allow_blank_text=True,
        ),
        original_price_by_category_condition=_marketplace_price_index(
            df,
            MARKETPLACE_CATEGORY_CONDITION_FIELDS,
            allow_blank_text=True,
        ),
        original_price_by_category=_marketplace_price_index(
            df,
            MARKETPLACE_CATEGORY_FIELDS,
            allow_blank_text=True,
        ),
        canonical_brand_by_key=canonical_brand_by_key,
        canonical_model_by_brand_model_key=canonical_model_by_brand_model_key,
        model_names_by_brand_key={
            brand_key: sorted(model_names)
            for brand_key, model_names in models_by_brand_key.items()
        },
        global_original_price=float(df["original_price"].median()),
        raw_rows=raw_rows,
    )


def _catalog_price_to_egp(original_price: Any, currency: Any) -> Optional[float]:
    price = pd.to_numeric(original_price, errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return None
    currency_text = _normalize_marketplace_text(currency, allow_blank=True).upper() or "EGP"
    if currency_text == "USD":
        return float(price) * _usd_to_egp_rate()
    if currency_text == "EGP":
        return float(price)
    return None


def _catalog_aliases(raw_aliases: Any) -> List[str]:
    text = _normalize_marketplace_text(raw_aliases, allow_blank=True)
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]


def _catalog_year(raw_year: Any) -> Optional[int]:
    year = pd.to_numeric(raw_year, errors="coerce")
    if pd.isna(year):
        return None
    return int(year)


def _catalog_reference_price_to_egp(raw_price: Any) -> Optional[float]:
    price = pd.to_numeric(raw_price, errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return None
    return float(price)


def _add_catalog_record(
    *,
    records_by_category_brand_model: Dict[tuple[Any, ...], List[CatalogPriceRecord]],
    records_by_brand_model: Dict[tuple[Any, ...], List[CatalogPriceRecord]],
    canonical_brand_by_key: Dict[str, str],
    canonical_model_by_brand_model_key: Dict[tuple[str, str], str],
    category: str,
    brand: str,
    model_name: str,
    original_price: float,
    production_year: Optional[int],
    aliases: List[str],
) -> None:
    brand_key = _marketplace_text_key(brand)
    canonical_brand_by_key.setdefault(brand_key, brand)
    record = CatalogPriceRecord(
        category=category,
        brand=brand,
        model=model_name,
        original_price=float(original_price),
        production_year=production_year,
    )

    candidate_models: List[str] = []
    for listed_model in [model_name, *aliases]:
        candidate_models.extend(_catalog_model_candidates(listed_model))

    for candidate_model in candidate_models:
        model_key = _marketplace_text_key(candidate_model)
        canonical_model_by_brand_model_key.setdefault((brand_key, model_key), model_name)
        category_brand_model_key = _marketplace_match_key_from_attrs(
            {"category": category, "brand": brand, "model": candidate_model},
            MARKETPLACE_CATEGORY_BRAND_MODEL_FIELDS,
            allow_blank_text=True,
        )
        brand_model_key = _marketplace_match_key_from_attrs(
            {"brand": brand, "model": candidate_model},
            MARKETPLACE_BRAND_MODEL_FIELDS,
            allow_blank_text=True,
        )
        records_by_category_brand_model.setdefault(category_brand_model_key, []).append(record)
        records_by_brand_model.setdefault(brand_model_key, []).append(record)


def _load_tennis_product_catalog(csv_path: Path) -> TennisProductCatalog:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tennis product catalog not found: {csv_path}")

    df = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    records_by_category_brand_model: Dict[tuple[Any, ...], List[CatalogPriceRecord]] = {}
    records_by_brand_model: Dict[tuple[Any, ...], List[CatalogPriceRecord]] = {}
    canonical_brand_by_key: Dict[str, str] = {}
    canonical_model_by_brand_model_key: Dict[tuple[str, str], str] = {}
    raw_rows = int(len(df))

    if set(TENNIS_PRODUCT_REFERENCE_FIELDS).issubset(set(df.columns)):
        for record in df.to_dict(orient="records"):
            original_price_egp = _catalog_reference_price_to_egp(record["egypt_original_price_egp_est"])
            if original_price_egp is None:
                continue
            _add_catalog_record(
                records_by_category_brand_model=records_by_category_brand_model,
                records_by_brand_model=records_by_brand_model,
                canonical_brand_by_key=canonical_brand_by_key,
                canonical_model_by_brand_model_key=canonical_model_by_brand_model_key,
                category=_normalize_marketplace_text(record["category"]),
                brand=_normalize_marketplace_text(record["brand"]),
                model_name=_normalize_marketplace_text(record["model_name"]),
                original_price=original_price_egp,
                production_year=_catalog_year(record["production_year"]),
                aliases=_catalog_aliases(record["aliases_for_fuzzy_matching"]),
            )
        return TennisProductCatalog(
            records_by_category_brand_model=records_by_category_brand_model,
            records_by_brand_model=records_by_brand_model,
            canonical_brand_by_key=canonical_brand_by_key,
            canonical_model_by_brand_model_key=canonical_model_by_brand_model_key,
            raw_rows=raw_rows,
            schema_name="tennis_market_product_catalog_reference_2018_2026",
        )

    if set(TENNIS_PRODUCT_CATALOG_FIELDS).issubset(set(df.columns)):
        for record in df.to_dict(orient="records"):
            original_price_egp = _catalog_price_to_egp(record["original_price"], record["currency"])
            if original_price_egp is None:
                continue
            _add_catalog_record(
                records_by_category_brand_model=records_by_category_brand_model,
                records_by_brand_model=records_by_brand_model,
                canonical_brand_by_key=canonical_brand_by_key,
                canonical_model_by_brand_model_key=canonical_model_by_brand_model_key,
                category=_normalize_marketplace_text(record["category"]),
                brand=_normalize_marketplace_text(record["brand"]),
                model_name=_normalize_marketplace_text(record["model"]),
                original_price=original_price_egp,
                production_year=None,
                aliases=_catalog_aliases(record["aliases"]),
            )
        return TennisProductCatalog(
            records_by_category_brand_model=records_by_category_brand_model,
            records_by_brand_model=records_by_brand_model,
            canonical_brand_by_key=canonical_brand_by_key,
            canonical_model_by_brand_model_key=canonical_model_by_brand_model_key,
            raw_rows=raw_rows,
            schema_name="simple_tennis_product_catalog",
        )

    raise ValueError(
        f"{csv_path} must match either the simple catalog schema or the 2018-2026 reference catalog schema."
    )


PRICE_DATASET_CSV_PATH: Optional[Path] = None
MARKETPLACE_PRICE_CATALOG: Optional[MarketplacePriceCatalog] = None
TENNIS_PRODUCT_CATALOG: Optional[TennisProductCatalog] = None
MARKETPLACE_ORIGINAL_PRICE_BY_ATTR: Dict[tuple[Any, ...], float] = {}
MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY_BRAND_MODEL: Dict[tuple[Any, ...], float] = {}
MARKETPLACE_ORIGINAL_PRICE_BY_BRAND_MODEL: Dict[tuple[Any, ...], float] = {}
MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY_CONDITION: Dict[tuple[Any, ...], float] = {}
MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY: Dict[tuple[Any, ...], float] = {}
MARKETPLACE_GLOBAL_ORIGINAL_PRICE: Optional[float] = None
MARKETPLACE_CSV_RAW_ROWS: int = 0
MARKETPLACE_LOOKUP_ROWS: int = 0
_MARKETPLACE_CATALOG_ERROR: Optional[str] = None
_TENNIS_PRODUCT_CATALOG_ERROR: Optional[str] = None

try:
    PRICE_DATASET_CSV_PATH = _resolve_price_training_csv_path(price_bundle)
    MARKETPLACE_PRICE_CATALOG = _load_marketplace_original_prices_from_training_csv(PRICE_DATASET_CSV_PATH)
    MARKETPLACE_ORIGINAL_PRICE_BY_ATTR = MARKETPLACE_PRICE_CATALOG.original_price_by_attr
    MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY_BRAND_MODEL = (
        MARKETPLACE_PRICE_CATALOG.original_price_by_category_brand_model
    )
    MARKETPLACE_ORIGINAL_PRICE_BY_BRAND_MODEL = MARKETPLACE_PRICE_CATALOG.original_price_by_brand_model
    MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY_CONDITION = MARKETPLACE_PRICE_CATALOG.original_price_by_category_condition
    MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY = MARKETPLACE_PRICE_CATALOG.original_price_by_category
    MARKETPLACE_GLOBAL_ORIGINAL_PRICE = MARKETPLACE_PRICE_CATALOG.global_original_price
    MARKETPLACE_CSV_RAW_ROWS = MARKETPLACE_PRICE_CATALOG.raw_rows
    MARKETPLACE_LOOKUP_ROWS = len(MARKETPLACE_ORIGINAL_PRICE_BY_ATTR)
except (FileNotFoundError, OSError, ValueError) as exc:
    _MARKETPLACE_CATALOG_ERROR = str(exc)

try:
    TENNIS_PRODUCT_CATALOG = _load_tennis_product_catalog(TENNIS_PRODUCT_CATALOG_CSV)
except (OSError, ValueError) as exc:
    _TENNIS_PRODUCT_CATALOG_ERROR = str(exc)

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
    condition: str = Field(..., examples=["New"])
    brand: str = Field(..., examples=["Wilson"])
    model: str = Field(..., examples=["Pro Staff 97 v14"])
    flaw: str = Field(..., examples=["None"])
    age_months: float = Field(..., ge=0, examples=[0])
    asking_price: Optional[float] = Field(
        None,
        ge=0,
        description="Optional seller asking price. If supplied, returns Underpriced/Fair/Overpriced.",
    )

    @field_validator("category", "condition", "brand", "model")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text fields cannot be empty")
        return value

    @field_validator("flaw")
    @classmethod
    def strip_flaw(cls, value: str) -> str:
        return value.strip()


class PriceBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[PriceItemFeatures] = Field(..., min_length=1, max_length=200)


def _price_label(asking_price: Optional[float], lower: float, upper: float) -> Optional[str]:
    if asking_price is None or float(asking_price) <= 0:
        return None
    if asking_price < lower:
        return "Underpriced"
    if asking_price > upper:
        return "Overpriced"
    return "Fair"


def _canonical_known_text(value: Any, candidates: List[str], *, allow_blank: bool = False) -> str:
    text = _normalize_marketplace_text(value, allow_blank=allow_blank)
    text_key = _marketplace_text_key(text, allow_blank=allow_blank)
    for candidate in candidates:
        if _marketplace_text_key(candidate, allow_blank=True) == text_key:
            return candidate
    return text


def _original_price_match(
    *,
    original_price: float,
    match_level: str,
    canonical_brand: str,
    canonical_model: str,
    dataset: Optional[str],
    is_estimated: bool = False,
    warnings: Optional[List[Dict[str, str]]] = None,
    catalog_year: Optional[int] = None,
    canonical_category: Optional[str] = None,
    canonical_condition: Optional[str] = None,
    canonical_flaw: Optional[str] = None,
) -> MarketplaceOriginalPriceMatch:
    return MarketplaceOriginalPriceMatch(
        original_price=float(original_price),
        match_level=match_level,
        canonical_brand=canonical_brand,
        canonical_model=canonical_model,
        dataset=dataset,
        is_estimated=is_estimated,
        warnings=warnings or [],
        catalog_year=catalog_year,
        canonical_category=canonical_category,
        canonical_condition=canonical_condition,
        canonical_flaw=canonical_flaw,
    )


def _marketplace_dataset_path_text() -> Optional[str]:
    return str(PRICE_DATASET_CSV_PATH) if PRICE_DATASET_CSV_PATH else None


def _tennis_catalog_path_text() -> Optional[str]:
    return str(TENNIS_PRODUCT_CATALOG_CSV) if TENNIS_PRODUCT_CATALOG_CSV.is_file() else None


def _inferred_product_year(age_months: Any) -> int:
    return date.today().year - int(float(age_months) // 12)


def _select_catalog_record(records: Optional[List[CatalogPriceRecord]], age_months: Any) -> Optional[CatalogPriceRecord]:
    if not records:
        return None
    target_year = _inferred_product_year(age_months)
    return min(
        records,
        key=lambda record: (
            1 if record.production_year is None else 0,
            abs((record.production_year or target_year) - target_year),
            -(record.production_year or 0),
        ),
    )


def _catalog_lookup_keys(attrs: Dict[str, Any], fields: List[str]) -> List[tuple[Any, ...]]:
    keys: List[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for model_candidate in _catalog_model_candidates(attrs["model"]):
        candidate_attrs = {**attrs, "model": model_candidate}
        key = _marketplace_match_key_from_attrs(candidate_attrs, fields, allow_blank_text=True)
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _select_catalog_record_for_keys(
    records_by_key: Dict[tuple[Any, ...], List[CatalogPriceRecord]],
    keys: List[tuple[Any, ...]],
    age_months: Any,
) -> Optional[CatalogPriceRecord]:
    for key in keys:
        catalog_record = _select_catalog_record(records_by_key.get(key), age_months)
        if catalog_record is not None:
            return catalog_record
    return None


def _canonical_marketplace_brand_model(attrs: Dict[str, Any]) -> tuple[str, str, str, str]:
    brand = _normalize_marketplace_text(attrs["brand"])
    model_name = _normalize_marketplace_text(attrs["model"])
    brand_key = _marketplace_text_key(brand)
    model_keys = [_marketplace_text_key(candidate) for candidate in _catalog_model_candidates(model_name)]
    canonical_brand = brand
    canonical_model = model_name
    if MARKETPLACE_PRICE_CATALOG is not None:
        canonical_brand = MARKETPLACE_PRICE_CATALOG.canonical_brand_by_key.get(brand_key, canonical_brand)
        for model_key in model_keys:
            canonical_model = MARKETPLACE_PRICE_CATALOG.canonical_model_by_brand_model_key.get(
                (brand_key, model_key),
                canonical_model,
            )
            if canonical_model != model_name:
                break
    if TENNIS_PRODUCT_CATALOG is not None:
        canonical_brand = TENNIS_PRODUCT_CATALOG.canonical_brand_by_key.get(brand_key, canonical_brand)
        for model_key in model_keys:
            catalog_model = TENNIS_PRODUCT_CATALOG.canonical_model_by_brand_model_key.get((brand_key, model_key))
            if catalog_model is not None:
                canonical_model = catalog_model
                break
    return brand_key, model_keys[0], canonical_brand, canonical_model


def _lookup_original_price_from_tennis_catalog(attrs: Dict[str, Any]) -> Optional[MarketplaceOriginalPriceMatch]:
    if TENNIS_PRODUCT_CATALOG is None or _TENNIS_PRODUCT_CATALOG_ERROR is not None:
        return None

    catalog_record = _select_catalog_record_for_keys(
        TENNIS_PRODUCT_CATALOG.records_by_category_brand_model,
        _catalog_lookup_keys(attrs, MARKETPLACE_CATEGORY_BRAND_MODEL_FIELDS),
        attrs["age_months"],
    )
    if catalog_record is not None:
        return _original_price_match(
            original_price=catalog_record.original_price,
            match_level="tennis_catalog_category_brand_model",
            canonical_brand=catalog_record.brand,
            canonical_model=catalog_record.model,
            dataset=_tennis_catalog_path_text(),
            catalog_year=catalog_record.production_year,
            canonical_category=catalog_record.category,
        )

    catalog_record = _select_catalog_record_for_keys(
        TENNIS_PRODUCT_CATALOG.records_by_brand_model,
        _catalog_lookup_keys(attrs, MARKETPLACE_BRAND_MODEL_FIELDS),
        attrs["age_months"],
    )
    if catalog_record is not None:
        return _original_price_match(
            original_price=catalog_record.original_price,
            match_level="tennis_catalog_brand_model",
            canonical_brand=catalog_record.brand,
            canonical_model=catalog_record.model,
            dataset=_tennis_catalog_path_text(),
            catalog_year=catalog_record.production_year,
            canonical_category=catalog_record.category,
        )

    return None


def _lookup_original_price_from_marketplace(item: PriceItemFeatures) -> MarketplaceOriginalPriceMatch:
    if _MARKETPLACE_CATALOG_ERROR is not None or MARKETPLACE_PRICE_CATALOG is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Marketplace training CSV is not available or failed to load; cannot resolve original_price. "
                f"{_MARKETPLACE_CATALOG_ERROR or 'Catalog is empty.'}"
            ),
        )
    attrs = item.model_dump()
    attrs.pop("asking_price", None)
    canonical_brand_model = _canonical_marketplace_brand_model(attrs)
    _, _, canonical_brand, canonical_model = canonical_brand_model

    catalog_match = _lookup_original_price_from_tennis_catalog(attrs)
    if catalog_match is not None:
        return catalog_match

    exact_key = _marketplace_match_key_from_attrs(attrs, MARKETPLACE_ATTR_FIELDS, allow_blank_text=True)
    original_price = MARKETPLACE_ORIGINAL_PRICE_BY_ATTR.get(exact_key)
    marketplace_dataset = _marketplace_dataset_path_text()
    if original_price is not None:
        return _original_price_match(
            original_price=original_price,
            match_level="exact_attributes",
            canonical_brand=canonical_brand,
            canonical_model=canonical_model,
            dataset=marketplace_dataset,
        )

    category_brand_model_key = _marketplace_match_key_from_attrs(
        attrs,
        MARKETPLACE_CATEGORY_BRAND_MODEL_FIELDS,
        allow_blank_text=True,
    )
    original_price = MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY_BRAND_MODEL.get(category_brand_model_key)
    if original_price is not None:
        return _original_price_match(
            original_price=original_price,
            match_level="category_brand_model_median",
            canonical_brand=canonical_brand,
            canonical_model=canonical_model,
            dataset=marketplace_dataset,
        )

    brand_model_key = _marketplace_match_key_from_attrs(
        attrs,
        MARKETPLACE_BRAND_MODEL_FIELDS,
        allow_blank_text=True,
    )
    original_price = MARKETPLACE_ORIGINAL_PRICE_BY_BRAND_MODEL.get(brand_model_key)
    if original_price is not None:
        return _original_price_match(
            original_price=original_price,
            match_level="brand_model_median",
            canonical_brand=canonical_brand,
            canonical_model=canonical_model,
            dataset=marketplace_dataset,
        )

    estimated_warning = _price_warning(
        "Brand/model was not found in the catalog; original_price was estimated from marketplace median data."
    )
    category_condition_key = _marketplace_match_key_from_attrs(
        attrs,
        MARKETPLACE_CATEGORY_CONDITION_FIELDS,
        allow_blank_text=True,
    )
    original_price = MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY_CONDITION.get(category_condition_key)
    if original_price is None:
        category_key = _marketplace_match_key_from_attrs(
            attrs,
            MARKETPLACE_CATEGORY_FIELDS,
            allow_blank_text=True,
        )
        original_price = MARKETPLACE_ORIGINAL_PRICE_BY_CATEGORY.get(category_key)
        if original_price is not None:
            return _original_price_match(
                original_price=original_price,
                match_level="fallback_category_median",
                canonical_brand=canonical_brand,
                canonical_model=canonical_model,
                dataset=marketplace_dataset,
                is_estimated=True,
                warnings=[estimated_warning],
            )

    if original_price is not None:
        return _original_price_match(
            original_price=original_price,
            match_level="fallback_category_condition_median",
            canonical_brand=canonical_brand,
            canonical_model=canonical_model,
            dataset=marketplace_dataset,
            is_estimated=True,
            warnings=[estimated_warning],
        )

    if MARKETPLACE_GLOBAL_ORIGINAL_PRICE is None:
        raise HTTPException(status_code=503, detail="Marketplace median original_price is not available.")

    return _original_price_match(
        original_price=MARKETPLACE_GLOBAL_ORIGINAL_PRICE,
        match_level="fallback_global_median",
        canonical_brand=canonical_brand,
        canonical_model=canonical_model,
        dataset=marketplace_dataset,
        is_estimated=True,
        warnings=[estimated_warning],
    )


def _price_feature_row(data: Dict[str, Any], marketplace_match: MarketplaceOriginalPriceMatch) -> Dict[str, Any]:
    categories = _price_metadata_categories()
    condition_values = ["New", "Like New", "Used"]
    flaw_values = ["", "None", "Minor", "Moderate", "Major"]
    feature_row = {
        **data,
        "category": marketplace_match.canonical_category
        or _canonical_known_text(data["category"], categories),
        "condition": marketplace_match.canonical_condition
        or _canonical_known_text(data["condition"], condition_values),
        "brand": marketplace_match.canonical_brand,
        "model": marketplace_match.canonical_model,
        "flaw": marketplace_match.canonical_flaw
        or _canonical_known_text(data["flaw"], flaw_values, allow_blank=True),
        "original_price": marketplace_match.original_price,
    }
    return feature_row


def _predict_price_one(item: PriceItemFeatures) -> Dict[str, Any]:
    data = item.model_dump()
    asking_price = data.pop("asking_price", None)
    marketplace_match = _lookup_original_price_from_marketplace(item)
    original_price = marketplace_match.original_price
    feature_row = _price_feature_row(data, marketplace_match)
    df = pd.DataFrame([{col: feature_row[col] for col in PRICE_FEATURE_COLS}])
    raw_predicted = float(price_model.predict(df)[0])
    max_price = _max_recommended_price(feature_row)
    predicted = min(max(0.0, raw_predicted), max_price)
    lower = max(0.0, predicted - PRICE_INTERVAL_HALF_WIDTH)
    upper = min(max_price, predicted + PRICE_INTERVAL_HALF_WIDTH)
    if upper < lower:
        lower = upper = predicted
    original_price_source = {
        "dataset": marketplace_match.dataset,
        "match_level": marketplace_match.match_level,
        "brand": marketplace_match.canonical_brand,
        "model": marketplace_match.canonical_model,
        "is_estimated": marketplace_match.is_estimated,
    }
    if marketplace_match.catalog_year is not None:
        original_price_source["catalog_year"] = marketplace_match.catalog_year
    response = {
        "recommended_price": _round_to_nearest_50_not_above(predicted, max_price),
        "raw_model_price_before_cap": round(raw_predicted, 2),
        "max_allowed_price": round(max_price, 2),
        "price_range": {"lower": round(lower, 2), "upper": round(upper, 2)},
        "currency": "EGP",
        "original_price": original_price,
        "original_price_source": original_price_source,
        "asking_price": asking_price,
        "label": _price_label(asking_price, lower, upper),
    }
    if marketplace_match.warnings:
        response["warnings"] = marketplace_match.warnings
    return response


@app.get("/price/health")
def price_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": PRICE_MODEL_PATH.exists(),
        "model_file": PRICE_MODEL_PATH.name,
        "feature_cols": PRICE_FEATURE_COLS,
    }


@app.get("/price/metadata")
def price_metadata() -> Dict[str, Any]:
    return {
        "feature_cols": PRICE_FEATURE_COLS,
        "model_file": PRICE_MODEL_PATH.name,
        "model_version": PRICE_METADATA.get("model_version"),
        "interval_half_width": PRICE_INTERVAL_HALF_WIDTH,
        "labeling_rule": PRICE_LABELING_RULE,
        "training_rows": price_bundle.get("training_rows"),
        "calibration_rows": price_bundle.get("calibration_rows"),
        "metadata": PRICE_METADATA,
        "marketplace_dataset_bundle_path": price_bundle.get("dataset_path"),
        "marketplace_dataset_csv_resolved": str(PRICE_DATASET_CSV_PATH) if PRICE_DATASET_CSV_PATH else None,
        "marketplace_dataset_csv_rows": MARKETPLACE_CSV_RAW_ROWS,
        "marketplace_lookup_rows": MARKETPLACE_LOOKUP_ROWS,
        "marketplace_brand_model_lookup_rows": len(MARKETPLACE_ORIGINAL_PRICE_BY_BRAND_MODEL),
        "marketplace_match_fields": MARKETPLACE_ATTR_FIELDS,
        "marketplace_fallback_match_fields": [
            MARKETPLACE_CATEGORY_BRAND_MODEL_FIELDS,
            MARKETPLACE_BRAND_MODEL_FIELDS,
            MARKETPLACE_CATEGORY_CONDITION_FIELDS,
            MARKETPLACE_CATEGORY_FIELDS,
            ["global_median"],
        ],
        "tennis_product_catalog_path": str(TENNIS_PRODUCT_CATALOG_CSV),
        "tennis_product_catalog_rows": TENNIS_PRODUCT_CATALOG.raw_rows if TENNIS_PRODUCT_CATALOG else 0,
        "tennis_product_catalog_schema": TENNIS_PRODUCT_CATALOG.schema_name if TENNIS_PRODUCT_CATALOG else None,
        "tennis_product_catalog_year_aware": (
            bool(TENNIS_PRODUCT_CATALOG and TENNIS_PRODUCT_CATALOG.schema_name != "simple_tennis_product_catalog")
        ),
        "tennis_product_catalog_ok": _TENNIS_PRODUCT_CATALOG_ERROR is None,
        "tennis_product_catalog_error": _TENNIS_PRODUCT_CATALOG_ERROR,
        "marketplace_known_brand_count": (
            len(MARKETPLACE_PRICE_CATALOG.canonical_brand_by_key) if MARKETPLACE_PRICE_CATALOG else 0
        ),
        "marketplace_catalog_ok": _MARKETPLACE_CATALOG_ERROR is None and bool(MARKETPLACE_ORIGINAL_PRICE_BY_ATTR),
        "marketplace_catalog_error": _MARKETPLACE_CATALOG_ERROR,
    }


@app.post("/price/predict")
def price_predict(item: PriceItemFeatures) -> Dict[str, Any]:
    try:
        return _predict_price_one(item)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Price prediction failed: {exc}") from exc


@app.post("/price/predict-batch")
def price_predict_batch(request: PriceBatchRequest) -> Dict[str, Any]:
    try:
        return {"results": [_predict_price_one(item) for item in request.items]}
    except HTTPException:
        raise
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
