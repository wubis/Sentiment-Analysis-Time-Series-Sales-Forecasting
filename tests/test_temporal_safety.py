import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sentiment_forecast.data import trends_table, reviews_table
from sentiment_forecast.features import (
    feature_matrix,
    feature_row,
    eligible_training_origins,
    issue_time,
    columns_for,
)
from sentiment_forecast.models import ridge_model
from sentiment_forecast.backtest import validate_config


def test_52_week_boundary_rejects_51_unavailable_windows():
    dates = pd.date_range("2020-01-05", periods=260, freq="W-SUN", tz="UTC")
    trends = trends_table(
        pd.DataFrame(
            {"date": dates, "available_at": dates + pd.Timedelta(days=1), "value": 50.0}
        )
    )
    features = pd.DataFrame({"target_lag_0": 50.0}, index=dates[15:])
    train, y = eligible_training_origins(
        trends, features, dates[188], 52, min_history_weeks=0
    )
    assert len(train) == 122
    assert 173 - len(train) == 51
    assert train.max() + pd.Timedelta(weeks=52) == dates[188]


def test_future_mutations_leave_features_scaler_and_prediction_unchanged(dataset):
    config, trends, reviews, coverage = dataset
    origin = trends.index[160]
    matrix = feature_matrix(trends, reviews, coverage, 4)
    changed_t = trends.copy()
    changed_t.loc[changed_t.index > origin, "value"] = 99.0
    changed_r = reviews.copy()
    changed_r.loc[changed_r.available_at > issue_time(origin), "text_score"] = -1.0
    changed_r.loc[changed_r.available_at > issue_time(origin), "rating"] = 1
    changed_c = coverage.copy()
    changed_c.loc[changed_c.index > origin, "complete"] = 0
    changed = feature_matrix(changed_t, changed_r, changed_c, 4)
    assert_frame_equal(matrix.loc[:origin], changed.loc[:origin])
    ix, y = eligible_training_origins(trends, matrix, origin, 4)
    ix2, y2 = eligible_training_origins(changed_t, changed, origin, 4)
    np.testing.assert_array_equal(ix, ix2)
    np.testing.assert_array_equal(y, y2)
    cols = columns_for(matrix, "E")
    first = ridge_model(10).fit(matrix.loc[ix, cols], y)
    second = ridge_model(10).fit(changed.loc[ix2, cols], y2)
    np.testing.assert_allclose(
        first["standardscaler"].mean_, second["standardscaler"].mean_
    )
    np.testing.assert_allclose(
        first.predict(matrix.loc[[origin], cols]),
        second.predict(changed.loc[[origin], cols]),
    )


def test_missing_target_is_not_filled_or_used_for_training(dataset):
    _, trends, reviews, coverage = dataset
    absent = trends.index[100]
    raw = trends.drop(absent).reset_index()
    grid = trends_table(raw)
    assert pd.isna(grid.loc[absent, "value"])
    features = feature_matrix(grid, reviews, coverage, 4)
    ix, _ = eligible_training_origins(grid, features, grid.index[160], 4)
    assert absent - pd.Timedelta(weeks=4) not in ix


def test_delayed_label_excluded_and_late_reviews_not_backfilled(dataset):
    _, trends, reviews, coverage = dataset
    origin = trends.index[150]
    delayed = trends.index[140]
    trends = trends.copy()
    trends.loc[delayed, "available_at"] = issue_time(origin) + pd.Timedelta(days=10)
    ix, _ = eligible_training_origins(
        trends, pd.DataFrame(index=trends.index), origin, 4
    )
    assert delayed - pd.Timedelta(weeks=4) not in ix
    empty = reviews.copy()
    empty["available_at"] = issue_time(origin) + pd.Timedelta(days=10)
    row = feature_row(trends, empty, coverage, origin, 4)
    assert np.isnan(row["text_mean_4"])
    assert row["volume_log_count_4"] == 0
    assert np.isnan(row["volume_age_weeks"])


def test_raw_input_guards(dataset):
    _, trends, reviews, coverage = dataset
    wrong = trends.reset_index()
    wrong["date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Sunday"):
        trends_table(wrong)
    wrong = reviews.copy()
    wrong.loc[0, "scorer_cutoff"] = wrong.loc[0, "published_at"]
    with pytest.raises(ValueError, match="Scorer"):
        reviews_table(wrong, "levis-505")
    wrong = trends.reset_index()
    wrong.loc[0, "available_at"] = wrong.loc[0, "date"]
    with pytest.raises(ValueError, match="closes"):
        trends_table(wrong)
    with pytest.raises(ValueError, match="exact product"):
        reviews_table(reviews, "505")


def test_selection_cannot_reach_test_future(dataset):
    config, trends, _, _ = dataset
    config["development_end"] = config["test_start"]
    with pytest.raises(ValueError):
        validate_config(config, trends)
    config["development_end"] = str(
        (pd.Timestamp(config["test_start"]) - pd.Timedelta(weeks=1)).date()
    )
    with pytest.raises(ValueError, match="maturation"):
        validate_config(config, trends)


def test_earliest_syndicated_copy_wins(dataset):
    _, _, reviews, _ = dataset
    duplicate = reviews.iloc[[0]].copy()
    duplicate["review_id"] = "future-copy"
    duplicate["available_at"] += pd.Timedelta(weeks=100)
    duplicate["text_score"] = -1.0
    output = reviews_table(pd.concat([reviews, duplicate]), "levis-505")
    assert len(output) == len(reviews)
    assert "future-copy" not in set(output.review_id)


def test_coverage_not_known_until_released(dataset):
    _, trends, reviews, coverage = dataset
    origin = trends.index[100]
    coverage = coverage.copy()
    coverage.loc[:, "available_at"] = issue_time(origin) + pd.Timedelta(days=1)
    assert feature_row(trends, reviews, coverage, origin, 4)["volume_coverage_4"] == 0


def test_fit_statistics_use_training_rows_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 2.0, 4.0]})
    model = ridge_model(1).fit(train, [1.0, 2.0, 3.0])
    model.predict(pd.DataFrame({"a": [1.0e12], "b": [1.0e12]}))
    np.testing.assert_allclose(model["simpleimputer"].statistics_, [2, 3])
    np.testing.assert_allclose(model["standardscaler"].mean_[:2], [2, 3])


def test_csv_preserves_identifier_leading_zeros(tmp_path):
    from sentiment_forecast.data import read_csv

    path = tmp_path / "identifiers.csv"
    path.write_text("review_id,duplicate_cluster,product_id\n001,002,005\n")
    row = read_csv(path).iloc[0]
    assert row.review_id == "001"
    assert row.duplicate_cluster == "002"
    assert row.product_id == "005"
