# Implementation status

## Implemented

The primary execution path is now the `sentiment_forecast` Python package. Root notebooks call it; original notebooks and outputs are archived with an explicit warning. The README no longer claims sales prediction or consistent sentiment improvements.

- Canonical weekly target/review/coverage schemas with explicit availability rules and missingness preservation.
- Exact product matching and earliest-copy duplicate handling.
- As-of review aggregation, target lag availability checks, and per-horizon training-label eligibility.
- Training-only imputation/scaling and matched ridge A–E ablations, with naive, seasonal-naive, and drift references.
- Chronological development tuning; a maturation gap before final origins; frozen hyperparameters during expanding final refits.
- Frozen temporal rating-proxy scorers: tested TF–IDF/ridge and an optional BERT backend with validation checkpoint selection and sample-weighted training loss.
- Forecast, fit, tuning, and model artifacts; reproducibility manifests; per-horizon metrics; paired block-bootstrap comparisons that withhold intervals when temporal support is inadequate.
- A reproducible synthetic fixture and regression tests for the original leakage mechanisms. The calendar now accepts both Sunday- and Saturday-ending weeks, and the supplied Trends export has a documented Saturday-ending conversion with assumed timing.

## Remaining empirical work

Some historical files have been added and audited; see [the data inventory](data-inventory.md), [review date audit](review-date-audit.md), and [product ID audit](product-mapping-audit.md). Collection logs, exact review dates, verified product mappings, label lineage, and Google Trends request metadata are still absent. No corrected empirical performance claim can be made until these are recovered and the documented inputs are supplied. The pipeline cannot certify manually asserted availability or scorer provenance.

The optional BERT branch requires extra dependencies/model downloads and has not been trained during this repair. Its shared preprocessing and temporal split logic are tested, but its GPU behavior and model quality remain unverified.

ETS/harmonic-ARIMA, gradient boosting, matched LSTMs, chronological scorer adaptation, ordinal/aspect models, prediction intervals, seasonal MASE, human sentiment annotation, platform/product transfer tests, and real-time Trends vintages remain proposed extensions. The initial implementation deliberately establishes the regularized direct-model experiment first. No sales/demand model or causal intervention analysis has been implemented.

Historical review/source imbalance, uncertain dates, and measurement error still require data-dependent analysis. Run manifests and test safeguards cannot replace data provenance or a prospectively held-out evaluation.

## Verification completed

On the local Python 3.11 environment (latest full run after the added-data audit):

- `python -m pytest -q`: **24 passed**, covering multi-step boundary overlap, future-data invariance, delayed labels, missing targets, training-only preprocessing, temporal scorer evaluation, scorer independence from test ratings, duplicate handling, stable identifiers, and end-to-end artifact generation.
- `ruff check src tests` and `ruff format --check src tests`: passed.
- `python -m pip check`: passed; editable package installation and CLI entry point verified.
- Full synthetic study: **1,827 forecast records** generated, including selected development forecasts and final-period forecasts. This is a software integration check, not evidence of predictive usefulness.
- Replacement notebook code syntax and local documentation links verified. The original data-dependent notebooks were not retrained.
