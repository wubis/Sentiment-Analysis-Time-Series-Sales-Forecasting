import numpy as np
import pandas as pd


def summarize(ledger):
    rows = []
    for (phase, model, horizon), group in ledger.groupby(["phase", "model", "horizon"]):
        valid = group.dropna(subset=["actual", "prediction"])
        errors = valid.prediction - valid.actual
        rows.append(
            dict(
                phase=phase,
                model=model,
                horizon=int(horizon),
                forecasts=len(group),
                scored=len(valid),
                mae=float(errors.abs().mean()),
                rmse=float(np.sqrt((errors**2).mean())),
            )
        )
    return pd.DataFrame(rows)


def paired_comparison(ledger, horizon=4, block_length=13, repetitions=2000, seed=42):
    """Negative E-C MAE difference favors text; few temporal blocks => no CI claim."""
    frame = ledger[(ledger.phase == "test") & (ledger.horizon == horizon)]
    frame = frame[frame.model.isin(["ridge_C", "ridge_E"])].copy()
    frame["loss"] = (frame.prediction - frame.actual).abs()
    pairs = frame.pivot(index="origin", columns="model", values="loss").dropna()
    result = {"horizon": horizon, "block_length": block_length, "seed": seed}
    if not {"ridge_C", "ridge_E"}.issubset(pairs.columns) or pairs.empty:
        return {**result, "status": "no paired observations"}
    differences = (pairs.ridge_E - pairs.ridge_C).to_numpy()
    result.update(
        n_origins=len(pairs), mae_difference_E_minus_C=float(differences.mean())
    )
    dates = pd.to_datetime(pairs.index, utc=True).sort_values()
    if (
        len(dates) > 1
        and not (dates.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all()
    ):
        return {
            **result,
            "status": "nonconsecutive origins; contiguous-block CI withheld",
        }
    if len(pairs) < 4 * block_length:
        return {**result, "status": "fewer than four dependence blocks; CI withheld"}
    rng = np.random.default_rng(seed)
    count = int(np.ceil(len(pairs) / block_length))
    samples = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(pairs) - block_length + 1, size=count)
        indices = np.concatenate([np.arange(s, s + block_length) for s in starts])[
            : len(pairs)
        ]
        samples.append(differences[indices].mean())
    result.update(
        status="descriptive paired moving-block bootstrap",
        ci95=[float(v) for v in np.quantile(samples, [0.025, 0.975])],
    )
    return result
