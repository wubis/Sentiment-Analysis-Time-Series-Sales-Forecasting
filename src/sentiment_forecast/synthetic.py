"""Deliberately synthetic fixture for testing; never a substitute for research data."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def write_fixture(directory, weeks=240, seed=17):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Fixture directory exists: {directory}")
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-05", periods=weeks, freq="W-SUN", tz="UTC")
    signal = np.sin(np.arange(weeks) * 2 * np.pi / 52)
    values = np.clip(50 + 15 * signal + rng.normal(0, 3, weeks), 0, 100)
    pd.DataFrame(
        {"date": dates, "available_at": dates + pd.Timedelta(days=1), "value": values}
    ).to_csv(directory / "trends.csv", index=False)
    reviews = []
    for i, date in enumerate(dates):
        for j in range(3):
            score = float(np.clip(0.4 * signal[i] + rng.normal(0, 0.3), -1, 1))
            published = date - pd.Timedelta(days=j + 1)
            reviews.append(
                dict(
                    review_id=f"{i}-{j}",
                    duplicate_cluster=f"{i}-{j}",
                    product_id="levis-505",
                    published_at=published,
                    available_at=published + pd.Timedelta(hours=12),
                    rating=int(np.clip(round(3 + 2 * score), 1, 5)),
                    text_score=score,
                    scorer_cutoff="2019-01-01T00:00:00Z",
                    scorer_id="SYNTHETIC-NOT-A-TRAINED-MODEL",
                )
            )
    pd.DataFrame(reviews).to_csv(directory / "reviews.csv", index=False)
    pd.DataFrame(
        {"date": dates, "available_at": dates + pd.Timedelta(days=1), "complete": 1}
    ).to_csv(directory / "coverage.csv", index=False)
    config = dict(
        product_id="levis-505",
        horizons=[1, 4, 13],
        alphas=[1.0, 10.0, 100.0],
        development_start=str(dates[130].date()),
        development_end=str(dates[150].date()),
        test_start=str(dates[164].date()),
        test_end=str(dates[-14].date()),
        min_train_examples=26,
        min_history_weeks=52,
        study_mode="synthetic",
        target_metadata=dict(
            query="SYNTHETIC Levi's 505",
            geography="SYNTHETIC",
            category="all",
            search_type="web",
            retrieved_at="synthetic generator",
            period_convention="Sunday-ending UTC",
        ),
    )
    (directory / "study.json").write_text(json.dumps(config, indent=2) + "\n")
    return directory / "study.json"
