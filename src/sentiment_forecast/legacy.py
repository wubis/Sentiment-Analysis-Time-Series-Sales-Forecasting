"""Reproducible, explicitly conditional preparation of the added legacy files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .data import trends_table

STYLE_CODES = (
    "501",
    "502",
    "505",
    "510",
    "511",
    "512",
    "514",
    "517",
    "527",
    "541",
    "550",
    "559",
    "711",
    "721",
    "724",
    "725",
    "726",
    "311",
    "312",
    "314",
    "315",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_trends(
    source: Path,
    output_csv: Path,
    output_manifest: Path,
    canonical_output: Path | None = None,
) -> dict:
    """Preserve source Sunday-start intervals with an explicit release assumption."""
    source = Path(source)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if (
        len(lines) < 4
        or not lines[0].startswith("Category:")
        or lines[2] != "Week,505 Levi: (United States)"
    ):
        raise ValueError(
            "Unexpected Google Trends preamble or query; inspect the source manually"
        )
    raw = pd.read_csv(source, skiprows=2)
    if raw.columns.tolist() != ["Week", "505 Levi: (United States)"]:
        raise ValueError("Unexpected Google Trends columns")
    dates = pd.to_datetime(raw.Week, format="%Y-%m-%d", errors="raise", utc=True)
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("Target dates must be sorted, unique, and nonmissing")
    if (dates.dt.dayofweek != 6).any() or (
        dates.diff().dropna() != pd.Timedelta(weeks=1)
    ).any():
        raise ValueError("The export must be consecutive Sunday-labeled weeks")
    values = pd.to_numeric(raw.iloc[:, 1], errors="raise")
    if values.isna().any() or not values.between(0, 100).all():
        raise ValueError("Trends index must contain complete numeric values in [0,100]")
    week_end_exclusive = dates + pd.Timedelta(weeks=1)
    assumed_available = week_end_exclusive + pd.Timedelta(days=1)
    file_modified = pd.Timestamp(source.stat().st_mtime, unit="s", tz="UTC")
    frame = pd.DataFrame(
        {
            "week_start": dates,
            "week_end_exclusive": week_end_exclusive,
            "value": values.astype(int),
            "assumed_first_available_at": assumed_available,
            "availability_basis": "ASSUMED_MONDAY_AFTER_SATURDAY_END",
            "possibly_partial_at_file_mtime": week_end_exclusive.gt(file_modified),
        }
    )
    output_csv = Path(output_csv)
    output_manifest = Path(output_manifest)
    canonical_output = (
        Path(canonical_output)
        if canonical_output
        else output_csv.with_name("trends_ASSUMED_saturday.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    canonical_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    complete = frame.loc[~frame.possibly_partial_at_file_mtime].copy()
    canonical = pd.DataFrame(
        {
            "date": complete.week_start + pd.Timedelta(days=6),
            "available_at": complete.assumed_first_available_at,
            "value": complete.value,
            "source_week_start": complete.week_start,
            "availability_basis": complete.availability_basis,
        }
    )
    trends_table(canonical, week_end_day="SAT")
    canonical.to_csv(canonical_output, index=False)
    manifest = {
        "source_file": str(source),
        "source_sha256": file_hash(source),
        "displayed_query": "505 Levi",
        "displayed_geography": "United States",
        "displayed_category": lines[0].removeprefix("Category: "),
        "search_type": "unknown",
        "retrieval_time": "unknown",
        "file_modified_at_utc": file_modified.isoformat(),
        "week_label_interpretation": "Sunday start, Saturday end; supported by published Google Trends research, source export does not spell out end dates",
        "available_at_interpretation": "ASSUMED Monday 00:00 UTC after the Saturday ending the source week; not observed publication or retrieval",
        "possibly_partial_at_file_mtime_count": int(
            frame.possibly_partial_at_file_mtime.sum()
        ),
        "canonical_target": str(canonical_output),
        "canonical_target_sha256": file_hash(canonical_output),
        "canonical_target_rows": len(canonical),
        "canonical_target_period_convention": "Saturday-ending UTC: source Sunday through Saturday; label is Saturday 00 UTC",
        "canonical_target_issue_offset_days": 2,
        "pipeline_compatibility": "SATURDAY CONFIG ONLY: set week_end_day=SAT, issue_offset_days=2, period_convention=Saturday-ending UTC; availability remains assumed",
        "study_scope": "source interval audit and timing sensitivity only",
        "rows": len(frame),
        "first_week": str(dates.iloc[0].date()),
        "last_week": str(dates.iloc[-1].date()),
        "output_sha256": file_hash(output_csv),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _review_frame(old: Path, recent: Path) -> pd.DataFrame:
    frames = []
    for name, path in [("older", old), ("recent", recent)]:
        frame = pd.read_pickle(path)  # Trusted user-supplied local files only.
        required = {
            "unique_id",
            "website",
            "review date(months ago)",
            "title",
            "comment",
        }
        if required - set(frame):
            raise ValueError(f"{path}: missing {sorted(required - set(frame))}")
        frames.append(frame.assign(review_slice=name))
    result = pd.concat(frames, ignore_index=True)
    result["website"] = result.website.astype(str).str.lower()
    return result


def audit_product_ids(
    old: Path, recent: Path, early: Path, v3: Path, output_csv: Path
) -> pd.DataFrame:
    """Evidence table for manual mapping; never emits an approved product join."""
    reviews = _review_frame(Path(old), Path(recent))
    prior = pd.read_pickle(early)
    later = pd.read_pickle(v3)
    keys = prior.unique_id.astype(str).str.extract(r"^(Levis|Macys)_(\d+)$")
    prior = prior.assign(
        site=keys[0].str.lower(), local_id=pd.to_numeric(keys[1], errors="coerce")
    )
    if prior.site.isna().any() or prior.local_id.isna().any():
        raise ValueError("Unexpected early table product key")
    early_names = prior.groupby(["site", "local_id"]).product_name.agg(
        lambda x: sorted(set(x))
    )
    late_names = later.groupby("unique_id").product_name.agg(lambda x: sorted(set(x)))
    late_sources = later.groupby("unique_id").source.agg(lambda x: sorted(set(x)))
    rows = []
    for (site, local_id), group in reviews.groupby(["website", "unique_id"], sort=True):
        text = (group.title.fillna("") + " " + group.comment.fillna("")).str.lower()
        mentions = {
            code: int(text.str.contains(rf"(?<!\d){code}(?!\d)", regex=True).sum())
            for code in STYLE_CODES
        }
        top_code = max(STYLE_CODES, key=lambda code: mentions[code])
        top_count = mentions[top_code]
        old_names = early_names.get((site, local_id), [])
        new_names = late_names.get(local_id, [])
        source_codes = late_sources.get(local_id, [])
        # Global numeric IDs in v3 are a different key space. Show them only as a collision warning.
        old_style = sorted(
            set(
                re.findall(
                    r"(?<!\d)(?:" + "|".join(STYLE_CODES) + r")(?!\d)",
                    " ".join(old_names),
                )
            )
        )
        new_style = sorted(
            set(
                re.findall(
                    r"(?<!\d)(?:" + "|".join(STYLE_CODES) + r")(?!\d)",
                    " ".join(new_names),
                )
            )
        )
        rows.append(
            {
                "website": site,
                "review_local_id": int(local_id),
                "reviews_total": len(group),
                "reviews_older": int(group.review_slice.eq("older").sum()),
                "reviews_recent": int(group.review_slice.eq("recent").sum()),
                "early_prefixed_name": " | ".join(old_names),
                "early_prefixed_style": " | ".join(old_style),
                "v3_numeric_name_UNSAFE_JOIN": " | ".join(new_names),
                "v3_numeric_source_UNSAFE_JOIN": " | ".join(map(str, source_codes)),
                "v3_numeric_style_UNSAFE_JOIN": " | ".join(new_style),
                "text_top_style": top_code if top_count else "",
                "text_top_style_mentions": top_count,
                "text_top_style_share": round(top_count / len(group), 4),
                "text_505_mentions": mentions["505"],
                "text_501_mentions": mentions["501"],
                "mapping_status": "UNVERIFIED_REQUIRES_ORIGINAL_PRODUCT_URL_OR_EXPORT",
            }
        )
    audit = pd.DataFrame(rows).sort_values(["website", "review_local_id"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_csv, index=False)
    return audit


def review_date_summary(old: Path, recent: Path) -> dict:
    reviews = _review_frame(Path(old), Path(recent))
    ages = pd.to_numeric(reviews["review date(months ago)"], errors="raise")
    absolute_date_columns = sorted(
        set(reviews) & {"published_at", "review_timestamp", "review_date", "date"}
    )
    collection_date_columns = sorted(
        set(reviews)
        & {"scraped_at", "collected_at", "first_observed_at", "snapshot_at"}
    )
    return {
        "rows": len(reviews),
        "distinct_age_values": int(ages.nunique()),
        "fractional_ages": int((ages % 1 != 0).sum()),
        "ages_over_20_years": int((ages > 240).sum()),
        "age_min_months": float(ages.min()),
        "age_max_months": float(ages.max()),
        "old_file_modified_utc": pd.Timestamp(
            Path(old).stat().st_mtime, unit="s", tz="UTC"
        ).isoformat(),
        "recent_file_modified_utc": pd.Timestamp(
            Path(recent).stat().st_mtime, unit="s", tz="UTC"
        ).isoformat(),
        "absolute_review_timestamps_found": bool(absolute_date_columns),
        "absolute_date_columns": absolute_date_columns,
        "scrape_or_reference_date_found": bool(collection_date_columns),
        "collection_date_columns": collection_date_columns,
        "note": "Filesystem modification times are not evidence of original scrape or review publication times.",
    }
