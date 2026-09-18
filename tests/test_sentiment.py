import pandas as pd
import pytest
from sentiment_forecast.sentiment import (
    prepare_reviews,
    temporal_partitions,
    score_reviews,
)


def raw_reviews():
    dates = pd.date_range("2018-01-01", periods=30, freq="30D", tz="UTC")
    return pd.DataFrame(
        [
            dict(
                review_id=str(i),
                duplicate_cluster=str(i),
                product_id="levis-505",
                published_at=d,
                available_at=d + pd.Timedelta(days=1),
                rating=1 + i % 5,
                text=f"Jeans review unique{i} "
                + ("great fit" if i % 2 else "poor fit"),
            )
            for i, d in enumerate(dates)
        ]
    )


def test_temporal_groups_and_fixed_label_mapping():
    raw = raw_reviews()
    copy = raw.iloc[[0]].copy()
    copy["review_id"] = "copy"
    copy["published_at"] = raw.published_at.iloc[-1]
    copy["available_at"] = raw.available_at.iloc[-1]
    frame = prepare_reviews(pd.concat([raw, copy]))
    assert len(frame) == len(raw)
    assert frame.scaled_rating.iloc[0] == -1
    assert frame.scaled_rating.iloc[4] == 1
    train, val, test = temporal_partitions(
        frame, "2018-10-01", "2019-04-01", "2020-07-01"
    )
    assert train.available_at.max() < val.published_at.min()
    assert val.available_at.max() < test.published_at.min()
    assert not set(train.duplicate_cluster) & set(test.duplicate_cluster)


def test_tfidf_end_to_end_exports_no_training_reviews(tmp_path):
    path = tmp_path / "reviews.csv"
    raw_reviews().to_csv(path, index=False)
    report = score_reviews(
        path, "2018-10-01", "2019-04-01", "2020-07-01", tmp_path / "scores"
    )
    scores = pd.read_csv(tmp_path / "scores/scored_reviews.csv")
    assert (
        pd.to_datetime(scores.published_at, utc=True)
        .gt(pd.Timestamp("2019-04-01", tz="UTC"))
        .all()
    )
    assert scores.text_score.between(-1, 1).all()
    assert (tmp_path / "scores/scorer.joblib").is_file()
    assert report["counts"]["test"] > 0


def test_exact_duplicates_with_different_cluster_ids_are_removed():
    raw = raw_reviews()
    raw.loc[20, "text"] = raw.loc[0, "text"].upper()
    assert len(prepare_reviews(raw)) == len(raw) - 1


def test_unresolved_dates_and_empty_text_fail():
    raw = raw_reviews()
    raw.loc[0, "text"] = None
    with pytest.raises(ValueError, match="Empty"):
        prepare_reviews(raw)


def test_future_test_ratings_do_not_select_or_change_scorer(tmp_path):
    raw = raw_reviews()
    original = tmp_path / "original.csv"
    changed = tmp_path / "changed.csv"
    raw.to_csv(original, index=False)
    raw.loc[raw.published_at.gt(pd.Timestamp("2019-04-01", tz="UTC")), "rating"] = 5
    raw.to_csv(changed, index=False)
    first = score_reviews(
        original, "2018-10-01", "2019-04-01", "2020-07-01", tmp_path / "first"
    )
    second = score_reviews(
        changed, "2018-10-01", "2019-04-01", "2020-07-01", tmp_path / "second"
    )
    assert first["selection"] == second["selection"]
    a = pd.read_csv(tmp_path / "first/scored_reviews.csv")
    b = pd.read_csv(tmp_path / "second/scored_reviews.csv")
    pd.testing.assert_series_equal(a.text_score, b.text_score)
    pd.testing.assert_series_equal(a.scorer_id, b.scorer_id)
