# Price Prediction API

This document describes the Marketplace Price Prediction part of the TennisFinder ML API.

Base URL examples:

```text
Local: http://127.0.0.1:8000
Ngrok: https://YOUR-NGROK-URL
```

The price model predicts a recommended resale price in EGP. The client does not send `original_price`; the API resolves it from backend data.

Active price model:

```text
models/price_model_catalog_augmented_2026.joblib
```

Backend data files:

```text
data/marketplace_price_dataset_egypt_tennis_expanded_65000.csv
data/tennis_product_catalog.csv
```

## Endpoints

```http
GET  /price/health
GET  /price/metadata
POST /price/predict
POST /price/predict-batch
```

## Main Flow

1. The client sends item attributes to `POST /price/predict`.
2. The API validates the request body.
3. The API rejects `original_price` if the client sends it.
4. The API resolves `original_price` in this order:
   - 2018-2026 tennis product catalog exact or alias match: `category + brand + model`
   - 2018-2026 tennis product catalog exact or alias match: `brand + model`
   - marketplace exact match: `category + condition + brand + model + flaw + age_months`
   - marketplace median: `category + brand + model`
   - marketplace median: `brand + model`
   - marketplace fallback median: `category + condition`
   - marketplace fallback median: `category`
   - marketplace global median
5. Catalog rows are year-aware. The API infers a target production year from `age_months` and picks the closest catalog year.
6. If the brand/model is not found, the API still returns `200` using an estimated `original_price` and a warning.
7. The API passes these model features to `price_model_catalog_augmented_2026.joblib`:

```text
category
condition
brand
model
flaw
age_months
original_price
```

8. The model returns a raw prediction.
9. The API caps the prediction using `original_price`, `condition`, and `flaw`.
10. The API rounds only `recommended_price` to the nearest 50 EGP without going above the cap.
11. The API does not round `price_range.lower` or `price_range.upper`, but the upper bound cannot exceed the cap.
12. If `asking_price` was sent and greater than `0`, the API calculates `label` from the unrounded capped range:
    - `Underpriced`: `asking_price < lower`
    - `Overpriced`: `asking_price > upper`
    - `Fair`: inside the range
13. If `asking_price` was not sent or is `0`, `label` is `null`.

## POST /price/predict

Predicts a recommended marketplace price for one item.

### Request Body

```json
{
  "category": "Racket",
  "condition": "New",
  "brand": "Wilson",
  "model": "Pro Staff 97 v14",
  "flaw": "None",
  "age_months": 0,
  "asking_price": 9000
}
```

### Request Fields

| Field | Type | Required | Rules | Notes |
|---|---:|---:|---|---|
| `category` | string | yes | cannot be empty | Example: `Racket` |
| `condition` | string | yes | cannot be empty | Dataset values include `New`, `Like New`, `Used` |
| `brand` | string | yes | cannot be empty | Matched case-insensitively |
| `model` | string | yes | cannot be empty | Matched case-insensitively; catalog aliases are supported; common racket-size text like `size 27` is ignored for catalog lookup |
| `flaw` | string | yes | can be empty string | Dataset values include empty string, `None`, `Minor`, `Moderate`, `Major` |
| `age_months` | number | yes | must be `>= 0` | Item age in months |
| `asking_price` | number or null | no | must be `>= 0` if sent | Used only for label calculation; `0` is treated as not provided |

Important:

```text
original_price must NOT be sent by the client.
```

If the client sends `original_price`, the API returns validation error `422`.

### Success Response: Catalog Current/New Capped Match

Status:

```http
200 OK
```

Example:

```json
{
  "recommended_price": 18900,
  "raw_model_price_before_cap": 22688.68,
  "max_allowed_price": 18900.0,
  "price_range": {
    "lower": 18023.6,
    "upper": 18900.0
  },
  "currency": "EGP",
  "original_price": 18900.0,
  "original_price_source": {
    "dataset": "D:\\Graduation Project\\TennisFinder_ALLmodels_v2\\tennis-finder-ai\\data\\tennis_product_catalog.csv",
    "match_level": "tennis_catalog_category_brand_model",
    "brand": "Wilson",
    "model": "Wilson Pro Staff 97",
    "is_estimated": false,
    "catalog_year": 2026
  },
  "asking_price": 9000.0,
  "label": "Underpriced"
}
```

### Success Response: Catalog Match

Example request uses a product from `data/tennis_product_catalog.csv`:

```json
{
  "category": "Bag",
  "condition": "New",
  "brand": "Artengo",
  "model": "Tennis Bag Backpack",
  "flaw": "None",
  "age_months": 0
}
```

Example response:

```json
{
  "recommended_price": 2350,
  "raw_model_price_before_cap": 2571.48,
  "max_allowed_price": 2350.0,
  "price_range": {
    "lower": 1473.6,
    "upper": 2350.0
  },
  "currency": "EGP",
  "original_price": 2350.0,
  "original_price_source": {
    "dataset": "D:\\Graduation Project\\TennisFinder_ALLmodels_v2\\tennis-finder-ai\\data\\tennis_product_catalog.csv",
    "match_level": "tennis_catalog_category_brand_model",
    "brand": "Artengo",
    "model": "Artengo Tennis Bag Backpack",
    "is_estimated": false,
    "catalog_year": 2026
  },
  "asking_price": null,
  "label": null
}
```

### Success Response: Unknown Brand/Model Fallback

Unknown brand/model requests do not fail. The API estimates `original_price` from marketplace medians.

Example response:

```json
{
  "recommended_price": 12150,
  "raw_model_price_before_cap": 15979.79,
  "max_allowed_price": 12166.0,
  "price_range": {
    "lower": 11289.6,
    "upper": 12166.0
  },
  "currency": "EGP",
  "original_price": 12166.0,
  "original_price_source": {
    "dataset": "D:\\Graduation Project\\TennisFinder_ALLmodels_v2\\tennis-finder-ai\\data\\marketplace_price_dataset_egypt_tennis_expanded_65000.csv",
    "match_level": "fallback_category_condition_median",
    "brand": "NeverBrand",
    "model": "Moonshot 999",
    "is_estimated": true
  },
  "asking_price": 9000.0,
  "label": "Underpriced",
  "warnings": [
    {
      "code": "ORIGINAL_PRICE_ESTIMATED",
      "message": "Brand/model was not found in the catalog; original_price was estimated from marketplace median data."
    }
  ]
}
```

### Success Response Fields

| Field | Type | Meaning |
|---|---:|---|
| `recommended_price` | integer | Capped recommended resale price rounded to nearest 50 EGP without exceeding `max_allowed_price` |
| `raw_model_price_before_cap` | number | Raw ML model output before marketplace business-rule caps |
| `max_allowed_price` | number | Maximum recommendation allowed from `original_price`, `condition`, and `flaw` |
| `price_range.lower` | number | Lower fair-price bound, not rounded to nearest 50 |
| `price_range.upper` | number | Upper fair-price bound, not rounded to nearest 50 and never above `max_allowed_price` |
| `currency` | string | Always `EGP` |
| `original_price` | number | Original item price fetched or estimated by the backend |
| `original_price_source.dataset` | string or null | Source CSV path |
| `original_price_source.match_level` | string | How `original_price` was resolved |
| `original_price_source.brand` | string | Canonical or submitted brand |
| `original_price_source.model` | string | Canonical or submitted model |
| `original_price_source.is_estimated` | boolean | `true` only when marketplace fallback estimation was used |
| `original_price_source.catalog_year` | integer | Present only for year-aware product catalog matches |
| `asking_price` | number or null | Seller price from request |
| `label` | string or null | `Underpriced`, `Fair`, `Overpriced`, or `null` |
| `warnings` | array | Only present when fallback estimation is used |

### `original_price_source.match_level` Values

| Value | Meaning |
|---|---|
| `exact_attributes` | Marketplace exact match for `category`, `condition`, `brand`, `model`, `flaw`, `age_months` |
| `category_brand_model_median` | Marketplace median for `category + brand + model` |
| `brand_model_median` | Marketplace median for `brand + model` |
| `tennis_catalog_category_brand_model` | Product catalog match for `category + brand + model` or model alias, closest to inferred production year |
| `tennis_catalog_brand_model` | Product catalog match for `brand + model` or model alias, closest to inferred production year |
| `fallback_category_condition_median` | Estimated from marketplace median for `category + condition` |
| `fallback_category_median` | Estimated from marketplace median for `category` |
| `fallback_global_median` | Estimated from marketplace global median |

### Pricing Cap Rules

The final recommendation cannot exceed `original_price * cap_ratio`.

| Condition | Flaw | Cap ratio |
|---|---|---:|
| `New` | `None` or empty | `1.00` |
| `Like New` | `None` or empty | `0.95` |
| `Like New` | `Minor` | `0.90` |
| `Used` | `None` or empty | `0.85` |
| `Used` | `Minor` | `0.78` |
| `Used` | `Moderate` | `0.65` |
| `Used` | `Major` | `0.50` |

Unknown condition/flaw combinations default to `0.80`.

## POST /price/predict Without Asking Price

Request:

```json
{
  "category": "Racket",
  "condition": "New",
  "brand": "Wilson",
  "model": "Pro Staff 97 v14",
  "flaw": "None",
  "age_months": 0
}
```

Response difference:

```json
{
  "asking_price": null,
  "label": null
}
```

The API still returns `recommended_price`, `price_range`, `currency`, `original_price`, and `original_price_source`.

## POST /price/predict-batch

Predicts prices for multiple items.

### Request Body

```json
{
  "items": [
    {
      "category": "Racket",
      "condition": "New",
      "brand": "Wilson",
      "model": "Pro Staff 97 v14",
      "flaw": "None",
      "age_months": 0,
      "asking_price": 9000
    },
    {
      "category": "Balls",
      "condition": "New",
      "brand": "Wilson",
      "model": "US Open 3-Pack",
      "flaw": "None",
      "age_months": 0
    }
  ]
}
```

### Request Rules

| Field | Type | Required | Rules |
|---|---:|---:|---|
| `items` | array | yes | minimum `1`, maximum `200` items |

Each item has the same schema as `POST /price/predict`.

### Success Response

Status:

```http
200 OK
```

Shape:

```json
{
  "results": [
    {
      "recommended_price": 15750,
      "price_range": {
        "lower": 14869.6,
        "upper": 16622.4
      },
      "currency": "EGP",
      "original_price": 15584.0,
      "original_price_source": {
        "dataset": "D:\\Graduation Project\\TennisFinder_ALLmodels_v2\\tennis-finder-ai\\data\\marketplace_price_dataset_egypt_tennis_expanded_65000.csv",
        "match_level": "exact_attributes",
        "brand": "Wilson",
        "model": "Pro Staff 97 v14",
        "is_estimated": false
      },
      "asking_price": 9000.0,
      "label": "Underpriced"
    }
  ]
}
```

Important: if any item has a validation error, the whole batch request fails.

## GET /price/health

Checks whether the price model file exists and returns model feature columns.

### Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_cols": [
    "category",
    "condition",
    "brand",
    "model",
    "flaw",
    "age_months",
    "original_price"
  ]
}
```

## GET /price/metadata

Returns price-model metadata and marketplace/catalog lookup status.

### Response Shape

```json
{
  "feature_cols": [
    "category",
    "condition",
    "brand",
    "model",
    "flaw",
    "age_months",
    "original_price"
  ],
  "interval_half_width": 876.4,
  "labeling_rule": "asking < lower => Underpriced; asking > upper => Overpriced; else Fair",
  "training_rows": 39000,
  "calibration_rows": 13000,
  "metadata": {
    "dataset_rows": 65000,
    "categories": [
      "Accessories",
      "Apparel",
      "Apparel Accessories",
      "Bag",
      "Ball Machine",
      "Balls",
      "Court Equipment",
      "Grips",
      "Overgrips",
      "Racket",
      "Shoes",
      "Stringing Machine",
      "Strings",
      "Training Aid"
    ],
    "brands": 34,
    "models": 392,
    "usd_to_egp_seed_rate": 53.1
  },
  "marketplace_dataset_bundle_path": "/mnt/data/marketplace_price_dataset_egypt_tennis_expanded_65000.csv",
  "marketplace_dataset_csv_resolved": "D:\\Graduation Project\\TennisFinder_ALLmodels_v2\\tennis-finder-ai\\data\\marketplace_price_dataset_egypt_tennis_expanded_65000.csv",
  "marketplace_dataset_csv_rows": 65000,
  "marketplace_lookup_rows": 27509,
  "marketplace_brand_model_lookup_rows": 459,
  "marketplace_match_fields": [
    "category",
    "condition",
    "brand",
    "model",
    "flaw",
    "age_months"
  ],
  "marketplace_fallback_match_fields": [
    ["category", "brand", "model"],
    ["brand", "model"],
    ["category", "condition"],
    ["category"],
    ["global_median"]
  ],
  "tennis_product_catalog_path": "D:\\Graduation Project\\TennisFinder_ALLmodels_v2\\tennis-finder-ai\\data\\tennis_product_catalog.csv",
  "tennis_product_catalog_rows": 4565,
  "tennis_product_catalog_schema": "tennis_market_product_catalog_reference_2018_2026",
  "tennis_product_catalog_year_aware": true,
  "tennis_product_catalog_ok": true,
  "tennis_product_catalog_error": null,
  "marketplace_known_brand_count": 33,
  "marketplace_catalog_ok": true,
  "marketplace_catalog_error": null
}
```

Some numeric values can change if the model bundle, marketplace dataset, or product catalog changes.

## Product Catalog

The inference-time catalog is stored at:

```text
data/tennis_product_catalog.csv
```

The current file comes from `tennis_market_product_catalog_reference_2018_2026_package.zip`.

Schema:

```text
record_id,category,subcategory,brand,product_line,variant,model_name,production_year,generation_tag,launch_year_est,end_year_est,tier,price_usd_msrp_est,egypt_original_price_egp_est,egypt_current_replacement_price_egp_est,egypt_direct_price_egp_if_seen,usd_egp_rate_used,egypt_import_retail_multiplier,price_source_type,confidence,global_source_url,egypt_source_url,aliases_for_fuzzy_matching,training_use_note,notes
```

Notes:

- The API uses `egypt_original_price_egp_est` as the `original_price` feature.
- The API uses `production_year` with `age_months` to choose the closest catalog year.
- `aliases_for_fuzzy_matching` is a semicolon-separated list of accepted model names.
- Alias matching is exact after trim + casefold; the API does not silently fuzzy-match to another product.
- Common racket length/size text in user input, such as `size 27`, `length 27`, or `27 inch`, is removed before catalog alias lookup.
- The package contains source and confidence fields for auditability.

## Error Responses

### Validation Error

Status:

```http
422 Unprocessable Entity
```

Client sends `original_price`:

```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": [
        "body",
        "original_price"
      ],
      "msg": "Extra inputs are not permitted",
      "input": 1
    }
  ]
}
```

Client sends negative `age_months`:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": [
        "body",
        "age_months"
      ],
      "msg": "Input should be greater than or equal to 0",
      "input": -1,
      "ctx": {
        "ge": 0
      }
    }
  ]
}
```

### Marketplace Dataset Not Available

Status:

```http
503 Service Unavailable
```

Response shape:

```json
{
  "detail": "Marketplace training CSV is not available or failed to load; cannot resolve original_price. <reason>"
}
```

This happens if the marketplace CSV used for median fallbacks cannot be found or cannot be loaded. Unknown brand/model is not a `503`; it returns `200` with fallback estimation.

### Prediction Failure

Status:

```http
400 Bad Request
```

Response shape:

```json
{
  "detail": "Price prediction failed: <reason>"
}
```

For batch:

```json
{
  "detail": "Price batch prediction failed: <reason>"
}
```

## Frontend Integration Notes

Send this:

```json
{
  "category": "Racket",
  "condition": "New",
  "brand": "Wilson",
  "model": "Pro Staff 97 v14",
  "flaw": "None",
  "age_months": 0,
  "asking_price": 9000
}
```

Do not send this:

```json
{
  "original_price": 15584
}
```

Recommended frontend behavior:

1. Use `/price/metadata` during debugging to confirm `marketplace_catalog_ok` is `true`.
2. Let the user enter/select `category`, `condition`, `brand`, `model`, `flaw`, `age_months`, and optionally `asking_price`.
3. Call `POST /price/predict`.
4. Display `recommended_price`, `price_range.lower`, `price_range.upper`, `currency`, and `label`.
5. If `warnings` exists, show a gentle note that the original price was estimated.
6. Treat `original_price` as backend/model context, not as a user-editable field.
