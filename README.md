# Review-augmented search-interest forecasting

This project tests whether review text adds predictive information for Levi’s 505 **Google Trends search interest**, beyond target history, seasonality, review volume, and star ratings. It does not currently measure sales or establish that sentiment improves forecasts.

The original notebook results were compromised by overlapping multi-step training/test labels, full-history scaling, test-driven early stopping, and an architecture-confounded comparison. See the [research audit and redesign](docs/research-redesign.md). The original notebooks and outputs are retained under [notebooks/archive](notebooks/archive/README.md) for audit only. Root notebooks now call the corrected package.

## What is implemented

- Explicit publication/availability timestamps, unique weekly target grids, exact product IDs, and duplicate-review handling.
- No target filling or future backfilling. Historical features use only records available at each issue time.
- Direct forecasts at 1, 4, and 13 weeks; training labels must have arrived at the current forecast origin.
- Training-only imputation/scaling, chronological development selection, frozen hyperparameters, and weekly test refits.
- Last-observation, seasonal-naive, and drift baselines; matched ridge A–E feature ablations.
- Temporal, deduplicated rating-proxy training using TF–IDF or optional BERT. Only post-selection reviews are exported for forecasting.
- Per-horizon MAE/RMSE, paired text-versus-rating comparisons, block-bootstrap safeguards, forecast/fit ledgers, saved fitted models, input/source hashes, and run manifests.

This is a **retrospective fixed-snapshot** pipeline. Historical Google Trends vintages and genuine source availability are needed before claiming a real-time deployment simulation. Some historical files have now been supplied and sorted in [the data inventory](docs/data-inventory.md). The Trends export has an [interval-preserving source table](data/prepared/target/trends_source_weeks.csv) and a [Saturday-ending canonical target with assumed timing](data/prepared/target/trends_ASSUMED_saturday.csv). The pipeline now supports that calendar when configured with `week_end_day: SAT` and `issue_offset_days: 2`. The [review date audit](docs/review-date-audit.md) and [product ID audit](docs/product-mapping-audit.md) found that exact review dates and verified product joins are still missing, so corrected real-data results cannot yet be reported.

## Install and test

Python 3.11 is the tested runtime. Create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

`requirements.lock` pins the tested core/test environment. Optional BERT is separately installed with `python -m pip install -e '.[bert]'`; it downloads `bert-base-uncased` and requires substantially more compute. The BERT training branch is not covered by the lightweight integration run. Record its environment separately for a research run.

## Run a complete synthetic software check

```bash
sentiment-forecast synthetic --output data/non_useful/synthetic
sentiment-forecast run \
  --config data/non_useful/synthetic/study.json \
  --trends data/non_useful/synthetic/trends.csv \
  --reviews data/non_useful/synthetic/reviews.csv \
  --coverage data/non_useful/synthetic/coverage.csv \
  --output artifacts/synthetic-001
```

Alternatively prefix `PYTHONPATH=src` and use `python -m sentiment_forecast.cli` in place of `sentiment-forecast`. Existing fixture/run directories are never overwritten; choose a new run name when repeating. Synthetic results validate software behavior, **not the research hypothesis**.

## Prepare real inputs

Start with [the data inventory](docs/data-inventory.md), [date audit](docs/review-date-audit.md), [product ID audit](docs/product-mapping-audit.md), and [data contract](docs/data-contract.md). Run `python scripts/prepare_added_data.py` to regenerate the audits and both Trends tables. The supplied Trends target is calendar-compatible after conversion; the study still needs exact review dates and product mappings, collection coverage, and the original Trends request/retrieval metadata. Use [the Saturday study example](configs/study.505-saturday.example.json) only after filling its provenance placeholders and supplying dated reviews. Do not feed the untraceable `avg_*sentiments` columns or relative “months ago” values into the new pipeline.

First fit a frozen scorer on a pre-period and evaluate it on later reviews. These dates are examples, not inferred dataset boundaries:

```bash
sentiment-forecast score-reviews \
  --reviews data/raw_reviews.csv \
  --train-end 2018-12-31T23:59:59Z \
  --validation-end 2019-06-30T23:59:59Z \
  --test-end 2019-12-31T23:59:59Z \
  --backend tfidf \
  --output artifacts/scorer-001
```

Use `artifacts/scorer-001/scored_reviews.csv` as the forecasting review input. Test labels never choose the scorer or its checkpoint. The target is explicitly `(rating - 3) / 2`; independent sentiment validity still requires human annotations.

Copy `configs/study.example.json` to `configs/study.json`, choose dates supported by the recovered data, and register that configuration before evaluating:

```bash
sentiment-forecast run \
  --config configs/study.json \
  --trends data/trends.csv \
  --reviews artifacts/scorer-001/scored_reviews.csv \
  --coverage data/coverage.csv \
  --output artifacts/research-001
```

Review `manifest.json`, `metrics.csv`, `forecasts.csv`, `fits.csv`, `tuning.csv`, `selected.json`, and `paired_comparisons.json` together. `models.joblib` stores fitted pipelines and training-origin manifests for final-period forecasts; only load artifacts you trust. Development errors have been used for model selection and are not unbiased final results. Only the predeclared evaluation period supports a final comparison; creating a new output directory does not make repeated peeking valid.

The primary comparison is `ridge_E` versus `ridge_C` at four weeks. Both use the same model family and hyperparameter budget:

| Variant | Inputs |
|---|---|
| A | Target lags and calendar seasonality |
| B | A + review volume and collection coverage |
| C | B + observed ratings |
| D | B + text-derived scores |
| E | C + text-derived scores |

See [implementation status](docs/implementation-status.md) for validation scope and remaining research work.
