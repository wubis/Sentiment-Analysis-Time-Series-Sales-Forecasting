"""Strict canonical inputs: UTC dates label configured Saturday- or Sunday-ending weeks."""

from pathlib import Path
import numpy as np
import pandas as pd


def timestamp(value):
    return pd.to_datetime(value, utc=True, errors="raise")


def require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input missing: {path}. See docs/data-contract.md; legacy derived averages are not accepted."
        )
    return pd.read_csv(
        path,
        dtype={
            name: "string"
            for name in ["review_id", "duplicate_cluster", "product_id", "scorer_id"]
        },
    )


def trends_table(frame, week_end_day="SUN"):
    require(frame, ["date", "available_at", "value"], "trends")
    frame = frame.copy()
    frame["date"] = timestamp(frame.date)
    frame["available_at"] = timestamp(frame.available_at)
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Trends dates must be nonmissing and unique")
    if week_end_day not in {"SUN", "SAT"}:
        raise ValueError("week_end_day must be SUN or SAT")
    expected_day = 6 if week_end_day == "SUN" else 5
    if (frame.date.dt.dayofweek != expected_day).any() or (
        frame.date != frame.date.dt.normalize()
    ).any():
        weekday_name = "Sunday" if week_end_day == "SUN" else "Saturday"
        raise ValueError(
            f"Trends dates must be {weekday_name} 00:00 UTC week-end labels; convert source period semantics explicitly"
        )
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    observed = frame.value.notna()
    if (
        not np.isfinite(frame.loc[observed, "value"]).all()
        or not frame.loc[observed, "value"].between(0, 100).all()
    ):
        raise ValueError("Observed Trends values must be finite and within [0, 100]")
    if frame.loc[observed, "available_at"].isna().any():
        raise ValueError("Observed targets require availability timestamps")
    if (
        frame.loc[observed, "available_at"]
        < frame.loc[observed, "date"] + pd.Timedelta(days=1)
    ).any():
        raise ValueError("Target cannot be available before its weekly period closes")
    if frame.empty:
        raise ValueError("Trends input is empty")
    frame = frame.sort_values("date").set_index("date")
    # Missing weeks remain missing, never interpolated or carried forward.
    return frame.reindex(
        pd.date_range(
            frame.index.min(), frame.index.max(), freq=f"W-{week_end_day}", tz="UTC"
        )
    ).rename_axis("date")


def reviews_table(frame, product_id):
    columns = [
        "review_id",
        "duplicate_cluster",
        "product_id",
        "published_at",
        "available_at",
        "rating",
        "text_score",
        "scorer_cutoff",
        "scorer_id",
    ]
    require(frame, columns, "reviews")
    frame = frame.copy()
    for col in ["review_id", "duplicate_cluster", "product_id", "scorer_id"]:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Reviews require nonempty {col}")
        frame[col] = frame[col].astype(str)
    if frame.review_id.duplicated().any():
        raise ValueError(
            "review_id must be unique; reconcile duplicate source records before ingestion"
        )
    for col in ["published_at", "available_at", "scorer_cutoff"]:
        frame[col] = timestamp(frame[col])
        if frame[col].isna().any():
            raise ValueError(f"Reviews require {col}")
    if (frame.available_at < frame.published_at).any():
        raise ValueError("Review availability cannot precede publication")
    # Strict pre-period scorer: all development/selection labels predate scored reviews.
    if (frame.scorer_cutoff >= frame.published_at).any():
        raise ValueError(
            "Scorer training/selection cutoff must precede every scored review's publication"
        )
    for col, lo, hi in [("rating", 1, 5), ("text_score", -1, 1)]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not frame[col].dropna().between(lo, hi).all():
            raise ValueError(f"{col} outside [{lo}, {hi}]")
    if frame.text_score.isna().any():
        raise ValueError("Every canonical review requires a text score")
    if not frame.rating.dropna().isin([1, 2, 3, 4, 5]).all():
        raise ValueError(
            "Review ratings must be integer stars, not pre-averaged ratings"
        )
    frame = frame[frame.product_id == str(product_id)]
    if frame.empty:
        raise ValueError(f"No reviews for exact product_id {product_id!r}")
    if frame.scorer_id.nunique() != 1 or frame.scorer_cutoff.nunique() != 1:
        raise ValueError(
            "This study requires one frozen pre-period scorer; mixed/adaptive scores are not supported"
        )
    # Earliest accessible copy wins; future syndicated copies cannot rewrite history.
    return (
        frame.sort_values(["available_at", "review_id"])
        .drop_duplicates("duplicate_cluster")
        .reset_index(drop=True)
    )


def coverage_table(frame, week_end_day="SUN"):
    require(frame, ["date", "available_at", "complete"], "coverage")
    frame = frame.copy()
    frame["date"] = timestamp(frame.date)
    frame["available_at"] = timestamp(frame.available_at)
    if (
        frame.date.isna().any()
        or frame.date.duplicated().any()
        or frame.available_at.isna().any()
    ):
        raise ValueError(
            "Coverage needs unique dates and nonmissing availability timestamps"
        )
    if week_end_day not in {"SUN", "SAT"}:
        raise ValueError("week_end_day must be SUN or SAT")
    expected_day = 6 if week_end_day == "SUN" else 5
    if (frame.date.dt.dayofweek != expected_day).any() or (
        frame.date != frame.date.dt.normalize()
    ).any():
        weekday_name = "Sunday" if week_end_day == "SUN" else "Saturday"
        raise ValueError(f"Coverage dates must use {weekday_name} weekly labels")
    if not frame.complete.isin([0, 1]).all():
        raise ValueError("Coverage complete must be 0 or 1")
    if (frame.available_at < frame.date + pd.Timedelta(days=1)).any():
        raise ValueError("Coverage cannot be certified before its week ends")
    return frame.set_index("date").sort_index()
