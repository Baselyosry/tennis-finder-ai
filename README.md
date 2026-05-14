# TennisFinder Combined ML API

This backend serves the three TennisFinder ML models from **one FastAPI app** and therefore **one ngrok URL**.

## Included models

1. Marketplace price recommendation
2. Court demand prediction
3. Smart matchmaking recommendation

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Model files (required before running)

The Git repository does **not** include the two largest weight files (they exceed GitHub’s size limits). After cloning, place these files next to the tracked small model:

- `models/price_model.joblib` (~96 MB)
- `models/matchmaking_model.joblib` (~198 MB)

`models/court_demand_model.joblib` is included in the repo. If you already have a full copy of this project (zip or another machine), copy the two `.joblib` files from `models/` into your clone’s `models/` folder.

**Marketplace price CSV:** `GET /price/metadata` includes `marketplace_dataset_bundle_path` (from `price_model.joblib`). Copy that training CSV into the repo using **only the file name** (for example `marketplace_price_dataset_egypt_tennis_expanded_65000.csv`) under **`models/`** or **`data/`**. Price prediction resolves `original_price` from that file; it is not read from the HTTP request.

## Run locally

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run with ngrok

```powershell
python run_ngrok.py
```

The script prints one public URL and all important endpoint links.

> Important: if an old single-model API is already running on the same ngrok reserved domain, stop it first before starting this combined API. After this combined API starts, the backend team can use one ngrok URL for all three models.

## Routes for backend team

### Shared

```http
GET /health
GET /docs
```

### Price model

```http
GET  /price/health
GET  /price/metadata
POST /price/predict
POST /price/predict-batch
```

Example request (send the item attributes; **`original_price` is not accepted** — it is resolved from the **same marketplace training CSV** referenced inside `price_model.joblib` (`dataset_path` basename), loaded from `models/<filename>.csv` or `data/<filename>.csv`. Lookup uses **exact match** on `category`, `condition`, `brand`, `model`, `flaw`, and `age_months` (when the training file has multiple rows for the same combo, the API uses the **median** `original_price`). If no combo matches, the API returns **404**.

```json
{
  "category": "Racket",
  "condition": "Used - Excellent",
  "brand": "Wilson",
  "model": "Pro Staff 97",
  "flaw": "None",
  "age_months": 12,
  "asking_price": 9000
}
```

### Court demand model

```http
GET  /court-demand/health
GET  /court-demand/metadata
POST /court-demand/predict
POST /court-demand/predict-batch
```

Example request:

```json
{
  "court_id": "C007",
  "court_type": "Hard",
  "location": "Giza",
  "hour": 9,
  "price": 157,
  "day_of_week": 2,
  "month": 2
}
```

### Smart matchmaking model

```http
GET  /matchmaking/health
GET  /matchmaking/metrics
GET  /matchmaking/id-mapping/status
GET  /matchmaking/id-mapping
GET  /matchmaking/id-mapping/by-internal/{internal_user_id}
GET  /matchmaking/id-mapping/by-public/{public_user_id}
POST /matchmaking/id-mapping/resolve
POST /matchmaking/id-mapping/sync
POST /matchmaking/players/sync
GET  /matchmaking/players
GET  /matchmaking/players/{user_id}
POST /matchmaking/predict-pair
POST /matchmaking/predict-features
GET  /matchmaking/recommend/{user_id}
POST /matchmaking/recommend-open-match
```

### Backend UUID migration without corruption

Use `POST /matchmaking/id-mapping/sync` to safely import backend UUIDs for existing players.
This endpoint is transactional (all-or-nothing) and blocks collisions by default.
`POST /matchmaking/players/sync` is kept as a backward-compatible alias.

Example sync request:

```json
{
  "mappings": [
    {
      "internal_user_id": 12,
      "public_user_id": "cd8de5f2-a8f4-4928-aa84-9e4ce71a607b"
    },
    {
      "internal_user_id": 33,
      "public_user_id": "6db67299-e7c7-4c92-a84d-3478a34ccb88"
    }
  ],
  "allow_reassign": false,
  "persist": true,
  "dry_run": false
}
```

Notes:
- Use `"dry_run": true` first to validate mapping files without applying.
- Keep `"allow_reassign": false` to prevent accidental remaps/corruption.
- With `"persist": true`, mappings are saved to `data/user_uuid_map.csv` and reloaded on restart.

Canonical mapping source for backend:
- Pull `GET /matchmaking/id-mapping` to get authoritative pairs of `internal_user_id` + `public_user_id`.
- Or use `POST /matchmaking/id-mapping/resolve` with a UUID list to resolve only known backend users.

Example pair prediction:

```json
{
  "user_id": "cd8de5f2-a8f4-4928-aa84-9e4ce71a607b",
  "candidate_id": "6db67299-e7c7-4c92-a84d-3478a34ccb88"
}
```

Example open match request:

```json
{
  "match_id": "7cb3ccaf-cc96-4270-8aa0-c71665f8c652",
  "user_id": "cd8de5f2-a8f4-4928-aa84-9e4ce71a607b",
  "candidates": [
    {"candidate_id": "6db67299-e7c7-4c92-a84d-3478a34ccb88"},
    {"candidate_id": "f4021d30-eac0-4db5-b5d8-ab95db04af72"},
    {"candidate_id": "27ad95fa-f726-45dc-ac3d-d51fd2f2dcc0"}
  ],
  "sort_by": "final_score"
}
```

## What to send the backend team

After running `python run_ngrok.py`, send them the printed base URL and this route map:

```text
Base URL: https://YOUR-NGROK-URL
Docs: https://YOUR-NGROK-URL/docs

Price: POST https://YOUR-NGROK-URL/price/predict
Court demand: POST https://YOUR-NGROK-URL/court-demand/predict
Matchmaking pair: POST https://YOUR-NGROK-URL/matchmaking/predict-pair
Open match recommendations: POST https://YOUR-NGROK-URL/matchmaking/recommend-open-match
```
