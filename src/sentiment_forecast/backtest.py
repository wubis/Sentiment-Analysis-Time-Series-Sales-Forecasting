"""Development selection followed by a frozen, scheduled-refit evaluation."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import timestamp, read_csv, trends_table, reviews_table, coverage_table
from .features import (
    VARIANTS,
    feature_matrix,
    columns_for,
    eligible_training_origins,
    issue_time,
)
from .models import ridge_model, baseline_predictions
from .metrics import summarize, paired_comparison


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config, trends):
    required = {
        "product_id",
        "horizons",
        "alphas",
        "development_start",
        "development_end",
        "test_start",
        "test_end",
        "min_train_examples",
        "min_history_weeks",
        "study_mode",
        "target_metadata",
    }
    if missing := required - set(config):
        raise ValueError(f"Study config missing {sorted(missing)}")
    if config["study_mode"] not in {"synthetic", "retrospective_fixed_snapshot"}:
        raise ValueError(
            "Only synthetic or retrospective_fixed_snapshot supported: real-time claims require vintage-aware targets"
        )
    metadata = config["target_metadata"]
    for key in [
        "query",
        "geography",
        "category",
        "search_type",
        "retrieved_at",
        "period_convention",
    ]:
        if not metadata.get(key):
            raise ValueError(f"target_metadata requires {key}")
    if metadata["period_convention"] != "Sunday-ending UTC":
        raise ValueError("Explicit conversion to Sunday-ending UTC is required")
    if any(str(value).startswith("REPLACE") for value in metadata.values()):
        raise ValueError("Replace example target metadata with audited source metadata")
    if config["study_mode"] != "synthetic" and pd.isna(
        timestamp(metadata["retrieved_at"])
    ):
        raise ValueError("A real study requires an actual target retrieval timestamp")
    horizons = config["horizons"]
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(type(h) is not int or not 1 <= h <= 52 for h in horizons)
    ):
        raise ValueError("horizons must be unique integers in [1, 52]")
    if 4 not in horizons:
        raise ValueError("The declared primary 4-week endpoint must be included")
    if not config["alphas"] or any(
        not np.isfinite(a) or a <= 0 for a in config["alphas"]
    ):
        raise ValueError("alphas must be finite positive numbers")
    if config["min_train_examples"] < 2 or config["min_history_weeks"] < 52:
        raise ValueError("Need at least two training examples and 52 weeks of history")
    dates = [
        timestamp(config[k])
        for k in ["development_start", "development_end", "test_start", "test_end"]
    ]
    if (
        any(d not in trends.index for d in dates)
        or not dates[0] <= dates[1] < dates[2] <= dates[3]
    ):
        raise ValueError(
            "Development/test dates must be ordered Sunday labels within the target grid"
        )
    if dates[1] + pd.Timedelta(weeks=max(horizons)) > dates[2]:
        raise ValueError(
            "Development labels extend beyond first test origin; leave a horizon-sized maturation gap"
        )
    scorer_start = config.get("scorer_warmup_end")
    if scorer_start and timestamp(scorer_start) >= dates[0]:
        raise ValueError("Scorer warmup must precede development")
    return dates


def run_study(config, trends, reviews, coverage, output_dir, input_hashes=None):
    dates = validate_config(config, trends)
    dev_start, dev_end, test_start, test_end = dates
    if reviews.scorer_cutoff.max() >= dev_start:
        raise ValueError(
            "Frozen scorer must be selected before forecasting development begins"
        )
    # Exclude target-training examples before the scorer exists, for every ablation.
    feature_start = max(
        trends.index.min() + pd.Timedelta(weeks=config["min_history_weeks"]),
        reviews.scorer_cutoff.max() + pd.Timedelta(weeks=13),
    )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Run directory already exists: {output}. Use a new run ID; evaluation results are not overwritten."
        )
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, tuning_rows, fit_rows, fitted_models = [], [], [], {}
    selected = {}
    try:
        for h in config["horizons"]:
            features = feature_matrix(trends, reviews, coverage, h)
            features = features.loc[features.index >= feature_start]
            candidates = {}
            dev_origins = trends.loc[dev_start:dev_end].index
            for variant in VARIANTS:
                cols = columns_for(features, variant)
                candidate_rows = {}
                for alpha in config["alphas"]:
                    predictions = []
                    for origin in dev_origins:
                        if origin not in features.index:
                            continue
                        target_date = origin + pd.Timedelta(weeks=h)
                        actual_row = trends.loc[target_date]
                        if pd.isna(actual_row.value) or pd.isna(
                            actual_row.available_at
                        ):
                            continue
                        if actual_row.available_at > issue_time(test_start):
                            raise ValueError(
                                "A development selection label is unavailable at first test issuance"
                            )
                        indices, y = eligible_training_origins(
                            trends, features, origin, h, config["min_history_weeks"]
                        )
                        if len(indices) < config["min_train_examples"]:
                            continue
                        model = ridge_model(alpha).fit(features.loc[indices, cols], y)
                        pred = float(model.predict(features.loc[[origin], cols])[0])
                        predictions.append(
                            dict(
                                phase="development",
                                model=f"ridge_{variant}",
                                origin=origin,
                                issued_at=issue_time(origin),
                                horizon=h,
                                target_date=target_date,
                                prediction=pred,
                                actual=float(actual_row.value),
                                alpha=alpha,
                            )
                        )
                    if not predictions:
                        raise ValueError(
                            f"No eligible development observations for h={h}; increase history or revise declared dates"
                        )
                    loss = float(
                        np.mean(
                            [abs(r["prediction"] - r["actual"]) for r in predictions]
                        )
                    )
                    tuning_rows.append(
                        dict(
                            variant=variant,
                            horizon=h,
                            alpha=alpha,
                            mae=loss,
                            n_origins=len(predictions),
                        )
                    )
                    candidate_rows[alpha] = predictions
                    candidates[(variant, alpha)] = loss
                best = min(
                    config["alphas"], key=lambda a: (candidates[(variant, a)], a)
                )
                selected[f"{variant}:{h}"] = best
                rows.extend(candidate_rows[best])
            # No further hyperparameter selection below this point for this horizon.
            for origin in trends.loc[test_start:test_end].index:
                if origin not in features.index:
                    raise ValueError("Insufficient post-scorer history at test origin")
                indices, y = eligible_training_origins(
                    trends, features, origin, h, config["min_history_weeks"]
                )
                if len(indices) < config["min_train_examples"]:
                    raise ValueError(
                        f"Insufficient training examples at {origin}, h={h}"
                    )
                target_date = origin + pd.Timedelta(weeks=h)
                actual = (
                    trends.loc[target_date, "value"]
                    if target_date in trends.index
                    else np.nan
                )
                common = dict(
                    phase="test",
                    origin=origin,
                    issued_at=issue_time(origin),
                    horizon=h,
                    target_date=target_date,
                    actual=actual,
                )
                for variant in VARIANTS:
                    alpha = selected[f"{variant}:{h}"]
                    cols = columns_for(features, variant)
                    model = ridge_model(alpha).fit(features.loc[indices, cols], y)
                    pred = float(model.predict(features.loc[[origin], cols])[0])
                    name = f"ridge_{variant}"
                    rows.append(
                        dict(**common, model=name, prediction=pred, alpha=alpha)
                    )
                    key = f"{name}/{h}/{origin.date()}"
                    fitted_models[key] = {
                        "pipeline": model,
                        "columns": cols,
                        "training_origins": indices,
                    }
                    labels = trends.reindex(indices + pd.Timedelta(weeks=h))
                    fit_rows.append(
                        dict(
                            model=name,
                            horizon=h,
                            origin=origin,
                            n_train=len(indices),
                            train_start=indices.min(),
                            train_end=indices.max(),
                            label_end=(indices + pd.Timedelta(weeks=h)).max(),
                            label_available_max=labels.available_at.max(),
                            alpha=alpha,
                        )
                    )
                for name, pred in baseline_predictions(
                    trends, origin, h, issue_time(origin)
                ).items():
                    rows.append(
                        dict(**common, model=name, prediction=pred, alpha=np.nan)
                    )
        ledger = pd.DataFrame(rows)
        ledger.to_csv(output / "forecasts.csv", index=False)
        summarize(ledger).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(output / "tuning.csv", index=False)
        pd.DataFrame(fit_rows).to_csv(output / "fits.csv", index=False)
        (output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
        comparisons = [
            paired_comparison(ledger, h, block_length=b)
            for h in config["horizons"]
            for b in sorted({max(h, 4), max(h, 13)})
        ]
        (output / "paired_comparisons.json").write_text(
            json.dumps(comparisons, indent=2) + "\n"
        )
        joblib.dump(fitted_models, output / "models.joblib", compress=3)
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        (output / "source").mkdir()
        for source in Path(__file__).parent.glob("*.py"):
            (output / "source" / source.name).write_bytes(source.read_bytes())
        manifest = dict(
            status="complete",
            study_mode=config["study_mode"],
            input_hashes=input_hashes or {},
            source_hashes=source_hashes,
            git_revision=revision,
            git_dirty=dirty,
            python=platform.python_version(),
            packages={
                p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]
            },
            selection="development MAE per variant/horizon; frozen during weekly test refits",
            primary="4-week MAE: ridge_E vs ridge_C",
            limitations=[
                "Fixed retrospective target snapshot; not a vintage-aware real-time simulation",
                "Development metrics are selected, not unbiased evaluation",
                "Synthetic results are software checks, not research evidence",
            ],
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return ledger
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps({"error": str(exc)}, indent=2) + "\n"
        )
        raise


def run_from_paths(config_path, trends_path, reviews_path, coverage_path, output):
    config = json.loads(Path(config_path).read_text())
    trends = trends_table(read_csv(trends_path))
    reviews = reviews_table(read_csv(reviews_path), config["product_id"])
    coverage = coverage_table(read_csv(coverage_path))
    paths = [config_path, trends_path, reviews_path, coverage_path]
    return run_study(
        config, trends, reviews, coverage, output, {str(p): sha256(p) for p in paths}
    )
