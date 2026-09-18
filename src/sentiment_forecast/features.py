"""Features materialized at historical issue times, never from future snapshots."""

import numpy as np
import pandas as pd

VARIANTS = {
    "A": ("target",),
    "B": ("target", "volume"),
    "C": ("target", "volume", "rating"),
    "D": ("target", "volume", "text"),
    "E": ("target", "volume", "rating", "text"),
}


def issue_time(origin, issue_offset_days=1):
    return origin + pd.Timedelta(days=issue_offset_days)


def feature_row(trends, reviews, coverage, origin, horizon, issue_offset_days=1):
    issue = issue_time(origin, issue_offset_days)
    row = {}
    for lag in (0, 1, 2, 3, 12, 51, 52):
        date = origin - pd.Timedelta(weeks=lag)
        value = np.nan
        if date in trends.index:
            observation = trends.loc[date]
            if pd.notna(observation.available_at) and observation.available_at <= issue:
                value = observation.value
        row[f"target_lag_{lag}"] = value
    target_date = origin + pd.Timedelta(weeks=horizon)
    phase = 2 * np.pi * (target_date.dayofyear - 1) / 365.2425
    for harmonic in (1, 2):
        row[f"target_sin_{harmonic}"] = np.sin(harmonic * phase)
        row[f"target_cos_{harmonic}"] = np.cos(harmonic * phase)
    eligible = reviews[(reviews.available_at <= issue) & (reviews.published_at < issue)]
    # Receipt-time aggregation avoids silently rewriting past weeks when old reviews arrive.
    for weeks in (4, 13):
        start = issue - pd.Timedelta(weeks=weeks)
        recent = eligible[eligible.available_at > start]
        dates = pd.date_range(origin - pd.Timedelta(weeks=weeks - 1), origin, freq="7D")
        known = coverage.reindex(dates)
        complete = known.complete.eq(1) & known.available_at.le(issue)
        row[f"volume_coverage_{weeks}"] = float(complete.mean())
        row[f"volume_log_count_{weeks}"] = np.log1p(len(recent))
        row[f"rating_mean_{weeks}"] = recent.rating.mean()
        row[f"rating_negative_share_{weeks}"] = (
            recent.rating.le(2).where(recent.rating.notna()).mean()
        )
        row[f"rating_count_{weeks}"] = np.log1p(recent.rating.notna().sum())
        row[f"text_mean_{weeks}"] = recent.text_score.mean()
        row[f"text_std_{weeks}"] = recent.text_score.std(ddof=0)
    row["volume_age_weeks"] = (
        (issue - eligible.available_at.max()).total_seconds() / (7 * 86400)
        if len(eligible)
        else np.nan
    )
    return row


def feature_matrix(trends, reviews, coverage, horizon, issue_offset_days=1):
    return pd.DataFrame(
        [
            feature_row(trends, reviews, coverage, date, horizon, issue_offset_days)
            for date in trends.index
        ],
        index=trends.index,
    )


def columns_for(matrix, variant):
    return [col for col in matrix if col.split("_", 1)[0] in VARIANTS[variant]]


def eligible_training_origins(
    trends, features, origin, horizon, min_history_weeks=52, issue_offset_days=1
):
    """Direct h-step label must have arrived by issue time, independently for each h."""
    label_dates = features.index + pd.Timedelta(weeks=horizon)
    labels = trends.reindex(label_dates)
    eligible = (
        (features.index >= trends.index.min() + pd.Timedelta(weeks=min_history_weeks))
        & (features.index < origin)
        & (label_dates <= origin)
        & labels.value.notna().to_numpy()
        & labels.available_at.le(issue_time(origin, issue_offset_days)).to_numpy()
    )
    return features.index[eligible], labels.value.to_numpy()[eligible]
