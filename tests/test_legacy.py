import json
import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from sentiment_forecast.legacy import (
    audit_product_ids,
    convert_trends,
    review_date_summary,
)
from sentiment_forecast.data import trends_table
from sentiment_forecast.features import issue_time


def test_trends_conversion_marks_assumed_availability(tmp_path):
    raw = tmp_path / "export.csv"
    raw.write_text(
        "Category: All categories\n\nWeek,505 Levi: (United States)\n"
        "2020-04-19,31\n2020-04-26,29\n2020-05-03,33\n"
    )
    target = tmp_path / "trends.csv"
    metadata = tmp_path / "assumptions.json"
    convert_trends(raw, target, metadata)
    frame = pd.read_csv(target)
    assert frame.assumed_first_available_at.tolist() == [
        "2020-04-27 00:00:00+00:00",
        "2020-05-04 00:00:00+00:00",
        "2020-05-11 00:00:00+00:00",
    ]
    assert frame.week_end_exclusive.iloc[0] == "2020-04-26 00:00:00+00:00"
    assert frame.availability_basis.eq("ASSUMED_MONDAY_AFTER_SATURDAY_END").all()
    assert frame.value.tolist() == [31, 29, 33]
    manifest = json.loads(metadata.read_text())
    assert manifest["retrieval_time"] == "unknown"
    assert "ASSUMED" in manifest["available_at_interpretation"]
    assert "SATURDAY CONFIG ONLY" in manifest["pipeline_compatibility"]
    assert len(manifest["source_sha256"]) == 64
    canonical = pd.read_csv(tmp_path / "trends_ASSUMED_saturday.csv")
    assert canonical.date.iloc[0] == "2020-04-25 00:00:00+00:00"
    assert canonical.available_at.iloc[0] == "2020-04-27 00:00:00+00:00"
    table = trends_table(canonical, "SAT")
    assert table.value.tolist() == [31, 29, 33]
    assert issue_time(table.index[0], 2) == pd.Timestamp("2020-04-27T00:00:00Z")


def test_possible_partial_source_week_is_excluded_from_canonical_target(tmp_path):
    raw = tmp_path / "export.csv"
    raw.write_text(
        "Category: All categories\n\nWeek,505 Levi: (United States)\n"
        "2020-04-19,31\n2020-04-26,29\n2020-05-03,33\n"
    )
    modified = datetime(2020, 5, 5, tzinfo=timezone.utc).timestamp()
    os.utime(raw, (modified, modified))
    manifest = convert_trends(
        raw, tmp_path / "source.csv", tmp_path / "assumptions.json"
    )
    assert manifest["possibly_partial_at_file_mtime_count"] == 1
    assert manifest["canonical_target_rows"] == 2
    assert pd.read_csv(tmp_path / "trends_ASSUMED_saturday.csv").value.tolist() == [
        31,
        29,
    ]


def test_trends_conversion_refuses_wrong_query_and_gap(tmp_path):
    raw = tmp_path / "export.csv"
    raw.write_text("Category: All categories\n\nWeek,other query\n2020-04-19,31\n")
    with pytest.raises(ValueError, match="preamble"):
        convert_trends(raw, tmp_path / "out.csv", tmp_path / "out.json")
    raw.write_text(
        "Category: All categories\n\nWeek,505 Levi: (United States)\n"
        "2020-04-19,31\n2020-05-03,33\n"
    )
    with pytest.raises(ValueError, match="consecutive"):
        convert_trends(raw, tmp_path / "out.csv", tmp_path / "out.json")


def test_site_aware_product_audit_detects_id_collision(tmp_path):
    older = tmp_path / "older.pkl"
    recent = tmp_path / "recent.pkl"
    reviews = pd.DataFrame(
        {
            "unique_id": [14, 41, 14],
            "website": ["levis", "macys", "levis"],
            "review date(months ago)": [24, 12, 1],
            "title": ["505 favorite", "501 Original", "505 jeans"],
            "comment": ["", "501 fit", ""],
        }
    )
    reviews.iloc[:2].to_pickle(older)
    reviews.iloc[2:].to_pickle(recent)
    early = tmp_path / "early.pkl"
    pd.DataFrame({"unique_id": ["Levis_14"], "product_name": ["511 Slim"]}).to_pickle(
        early
    )
    v3 = tmp_path / "v3.pkl"
    pd.DataFrame(
        {
            "unique_id": [14, 41],
            "product_name": ["311 Skinny", "505 Regular"],
            "source": [1, 1],
        }
    ).to_pickle(v3)
    audit = audit_product_ids(older, recent, early, v3, tmp_path / "audit.csv")
    levi = audit[(audit.website == "levis") & audit.review_local_id.eq(14)].iloc[0]
    macys = audit[(audit.website == "macys") & audit.review_local_id.eq(41)].iloc[0]
    assert levi.early_prefixed_name == "511 Slim"
    assert levi.v3_numeric_name_UNSAFE_JOIN == "311 Skinny"
    assert levi.text_top_style == "505"
    assert macys.early_prefixed_name == ""
    assert macys.v3_numeric_name_UNSAFE_JOIN == "505 Regular"
    assert macys.text_top_style == "501"
    assert audit.mapping_status.eq(
        "UNVERIFIED_REQUIRES_ORIGINAL_PRODUCT_URL_OR_EXPORT"
    ).all()
    assert (
        review_date_summary(older, recent)["absolute_review_timestamps_found"] is False
    )
