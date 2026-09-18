"""Inspect user-supplied legacy data and create conditional outputs.

Run from the repository root after `pip install -e .`.
"""

import json
from pathlib import Path
from sentiment_forecast.legacy import (
    convert_trends,
    audit_product_ids,
    review_date_summary,
)

root = Path(__file__).resolve().parents[1]
data = root / "data"
old = data / "useful/reviews/df_reviews_1year_ago.pkl"
recent = data / "useful/reviews/df_reviews_within_1year.pkl"
manifest = convert_trends(
    data / "useful/target/505_trend.csv",
    data / "prepared/target/trends_source_weeks.csv",
    data / "prepared/target/trends_assumptions.json",
    data / "prepared/target/trends_ASSUMED_saturday.csv",
)
audit = audit_product_ids(
    old,
    recent,
    data / "non_useful/legacy_processed/data_sentiment_merged.pkl",
    data / "non_useful/legacy_processed/data_sentiment_merged_v3.pkl",
    root / "docs/product-id-audit.csv",
)
summary = review_date_summary(old, recent)
(root / "docs/review-date-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    f"Prepared {manifest['rows']} source weeks and {manifest['canonical_target_rows']} complete Saturday-ending target weeks with assumed availability"
)
print(
    f"Audited {len(audit)} site/product-ID pairs; all require original source confirmation"
)
print(
    f"Review date summary: {summary['distinct_age_values']} distinct relative ages across {summary['rows']} rows"
)
