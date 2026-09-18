import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sentiment_forecast.backtest import run_study
from sentiment_forecast.metrics import summarize, paired_comparison


def small_config(config, trends):
    config = dict(config)
    config.update(
        horizons=[1, 4],
        alphas=[1.0, 10.0],
        development_start=str(trends.index[125].date()),
        development_end=str(trends.index[127].date()),
        test_start=str(trends.index[140].date()),
        test_end=str(trends.index[144].date()),
    )
    return config


def test_end_to_end_and_test_outcomes_do_not_select_hyperparameters(dataset, tmp_path):
    config, trends, reviews, coverage = dataset
    config = small_config(config, trends)
    first = run_study(config, trends, reviews, coverage, tmp_path / "first")
    changed = trends.copy()
    changed.loc[
        changed.index > pd.Timestamp(config["test_start"], tz="UTC"), "value"
    ] = 90.0
    second = run_study(config, changed, reviews, coverage, tmp_path / "second")
    assert json.loads((tmp_path / "first/selected.json").read_text()) == json.loads(
        (tmp_path / "second/selected.json").read_text()
    )
    origin = pd.Timestamp(config["test_start"], tz="UTC")
    cols = ["model", "horizon", "prediction"]
    assert_frame_equal(
        first[first.origin.eq(origin)][cols].reset_index(drop=True),
        second[second.origin.eq(origin)][cols].reset_index(drop=True),
    )
    fits = pd.read_csv(tmp_path / "first/fits.csv")
    assert (
        pd.to_datetime(fits.label_end, utc=True)
        <= pd.to_datetime(fits.origin, utc=True)
    ).all()
    assert (
        pd.to_datetime(fits.label_available_max, utc=True)
        <= pd.to_datetime(fits.origin, utc=True) + pd.Timedelta(days=1)
    ).all()
    assert ((first.target_date - first.origin).dt.days == first.horizon * 7).all()
    assert (tmp_path / "first/models.joblib").is_file()
    with pytest.raises(FileExistsError):
        run_study(config, trends, reviews, coverage, tmp_path / "first")


def test_development_label_reporting_delay_blocks_selection(dataset, tmp_path):
    config, trends, reviews, coverage = dataset
    config = small_config(config, trends)
    trends = trends.copy()
    trends.loc[trends.index[126], "available_at"] = trends.index[180]
    with pytest.raises(ValueError, match="selection label"):
        run_study(config, trends, reviews, coverage, tmp_path / "bad")
    assert (tmp_path / "bad/failure.json").is_file()
    assert not (tmp_path / "bad/manifest.json").exists()


def test_metrics_missing_actuals_and_small_sample_ci():
    ledger = pd.DataFrame(
        {
            "phase": ["test"] * 3,
            "model": ["naive"] * 3,
            "horizon": [4] * 3,
            "prediction": [1.0, 4.0, 10.0],
            "actual": [2.0, 2.0, np.nan],
        }
    )
    metrics = summarize(ledger).iloc[0]
    assert metrics.scored == 2 and metrics.forecasts == 3
    assert metrics.mae == 1.5
    assert metrics.rmse == pytest.approx(np.sqrt(2.5))
    rows = []
    for i, date in enumerate(pd.date_range("2020-01-05", periods=10, freq="W-SUN")):
        for name, pred in [("ridge_C", 2.0), ("ridge_E", 1.0)]:
            rows.append(
                dict(
                    phase="test",
                    model=name,
                    horizon=4,
                    origin=date,
                    prediction=pred,
                    actual=0.0,
                )
            )
    result = paired_comparison(pd.DataFrame(rows), block_length=4)
    assert result["mae_difference_E_minus_C"] == -1
    assert "ci95" not in result


def test_missing_real_data_fails_clearly(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"product_id":"levis-505"}')
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "sentiment_forecast.cli",
            "run",
            "--config",
            str(config),
            "--trends",
            str(tmp_path / "absent.csv"),
            "--reviews",
            "missing",
            "--coverage",
            "missing",
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "Required input missing" in process.stderr
    assert not (tmp_path / "run").exists()
