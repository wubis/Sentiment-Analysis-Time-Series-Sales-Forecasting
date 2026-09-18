"""Frozen, temporally evaluated rating-proxy scorers with disjoint review groups."""

import hashlib
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from .data import read_csv, require, timestamp
from .backtest import sha256


def prepare_reviews(frame):
    require(
        frame,
        [
            "review_id",
            "duplicate_cluster",
            "product_id",
            "published_at",
            "available_at",
            "rating",
            "text",
        ],
        "raw reviews",
    )
    frame = frame.copy()
    for col in ["review_id", "duplicate_cluster", "product_id"]:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Raw reviews require nonempty {col}")
        frame[col] = frame[col].astype(str)
    if frame.review_id.duplicated().any():
        raise ValueError("Raw review IDs must be unique")
    for col in ["published_at", "available_at"]:
        frame[col] = timestamp(frame[col])
        if frame[col].isna().any():
            raise ValueError(
                f"Raw reviews require {col}; rounded relative dates are not accepted"
            )
    if (frame.available_at < frame.published_at).any():
        raise ValueError("Review availability precedes publication")
    frame["text"] = (
        frame.text.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if frame.text.eq("").any():
        raise ValueError("Empty review text must be resolved before training")
    frame["rating"] = pd.to_numeric(frame.rating, errors="raise")
    if not frame.rating.dropna().isin([1, 2, 3, 4, 5]).all():
        raise ValueError(
            "Ratings must be integer stars from 1 to 5 or missing for inference"
        )
    frame["text_hash"] = frame.text.str.lower().map(
        lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    # Connected components combine curator near-duplicate clusters and exact-text matches.
    # Retain the earliest record of each component, never a later syndicated copy.
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in frame.itertuples():
        a, b = root("cluster:" + row.duplicate_cluster), root("text:" + row.text_hash)
        parent[a] = b
    frame["dedup_group"] = frame.duplicate_cluster.map(lambda s: root("cluster:" + s))
    frame = frame.sort_values(["available_at", "review_id"]).drop_duplicates(
        "dedup_group"
    )
    frame["scaled_rating"] = (
        frame.rating - 3
    ) / 2  # Explicit fixed mapping: 1 -> -1, 5 -> +1.
    return frame.reset_index(drop=True)


def temporal_partitions(frame, train_end, validation_end, test_end):
    train_end, validation_end, test_end = map(
        timestamp, [train_end, validation_end, test_end]
    )
    if not train_end < validation_end < test_end:
        raise ValueError("Require train_end < validation_end < test_end")
    train = frame[frame.available_at.le(train_end) & frame.rating.notna()]
    val = frame[
        frame.published_at.gt(train_end)
        & frame.available_at.le(validation_end)
        & frame.rating.notna()
    ]
    test = frame[
        frame.published_at.gt(validation_end)
        & frame.available_at.le(test_end)
        & frame.rating.notna()
    ]
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if len(part) < 2:
            raise ValueError(
                f"Need at least two independently dated labeled reviews in {name}"
            )
    for a, b in [(train, val), (train, test), (val, test)]:
        if set(a.duplicate_cluster) & set(b.duplicate_cluster) or set(
            a.text_hash
        ) & set(b.text_hash):
            raise ValueError("Duplicate reviews cross temporal partitions")
    return train, val, test


def regression_metrics(actual, prediction):
    error = np.asarray(prediction) - np.asarray(actual)
    return dict(
        n=len(error),
        mae=float(np.abs(error).mean()),
        mse=float((error**2).mean()),
        rmse=float(np.sqrt((error**2).mean())),
    )


def fit_tfidf(train, val, output, seed):
    candidates = []
    for alpha in (1.0, 10.0, 100.0):
        model = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True
            ),
            Ridge(alpha=alpha, solver="lsqr"),
        )
        model.fit(train.text, train.scaled_rating)
        pred = np.clip(model.predict(val.text), -1, 1)
        candidates.append(
            (regression_metrics(val.scaled_rating, pred)["mae"], alpha, model)
        )
    _, alpha, model = min(candidates, key=lambda row: (row[0], row[1]))
    joblib.dump(model, output / "scorer.joblib")
    return lambda texts: np.clip(model.predict(list(texts)), -1, 1), dict(
        alpha=alpha,
        validation_candidates=[{"alpha": a, "mae": loss} for loss, a, _ in candidates],
    )


def fit_bert(train, val, output, seed):
    """Optional backend; test reviews never control epoch selection."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ValueError(
            "BERT requires optional dependencies: pip install -e '.[bert]'"
        ) from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Save the downloaded tokenizer/config with the selected checkpoint for replay.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    batch_size, max_length = 16, 128
    generator = np.random.default_rng(seed)

    def predict(texts):
        texts = list(texts)
        results = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                results.extend(model(**batch).logits.squeeze(-1).cpu().numpy())
        return np.clip(np.asarray(results), -1, 1)

    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(5):
        model.train()
        losses = []
        order = generator.permutation(len(train))
        for start in range(0, len(train), batch_size):
            rows = train.iloc[order[start : start + batch_size]]
            batch = tokenizer(
                rows.text.tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(
                rows.scaled_rating.to_numpy(), dtype=torch.float32, device=device
            )
            optimizer.zero_grad()
            pred = model(**batch).logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((loss.item(), len(rows)))
        validation = regression_metrics(val.scaled_rating, predict(val.text))
        history.append(
            dict(
                epoch=epoch + 1,
                train_mse=sum(loss * n for loss, n in losses) / len(train),
                validation=validation,
            )
        )
        if validation["mae"] < best_loss:
            best_loss = validation["mae"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.save_pretrained(output / "bert")
    tokenizer.save_pretrained(output / "bert")
    return predict, dict(
        history=history,
        max_length=max_length,
        base_model="bert-base-uncased",
        resolved_revision=getattr(model.config, "_commit_hash", None),
        device=str(device),
    )


def score_reviews(
    path, train_end, validation_end, test_end, output_dir, backend="tfidf", seed=42
):
    frame = prepare_reviews(read_csv(path))
    train, val, test = temporal_partitions(frame, train_end, validation_end, test_end)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    output.mkdir(parents=True)
    if backend not in {"tfidf", "bert"}:
        raise ValueError("Unknown scorer backend")
    fit = fit_tfidf if backend == "tfidf" else fit_bert
    predict, details = fit(train, val, output, seed)
    test_pred = predict(test.text)
    baseline = np.full(len(test), train.scaled_rating.mean())
    report = dict(
        backend=backend,
        seed=seed,
        label="(rating - 3) / 2; rating proxy, not independent sentiment",
        train_end=str(timestamp(train_end)),
        validation_end=str(timestamp(validation_end)),
        test_end=str(timestamp(test_end)),
        counts={"train": len(train), "validation": len(val), "test": len(test)},
        test=regression_metrics(test.scaled_rating, test_pred),
        constant_baseline=regression_metrics(test.scaled_rating, baseline),
        by_rating={
            str(r): regression_metrics(
                test.loc[test.rating.eq(r), "scaled_rating"],
                test_pred[test.rating.eq(r).to_numpy()],
            )
            for r in sorted(test.rating.unique())
        },
        selection=details,
        input_sha256=sha256(path),
    )
    # Scorer identity depends on the frozen artifact, not future test outcomes.
    artifact_paths = (
        sorted((output / "bert").glob("*"))
        if backend == "bert"
        else [output / "scorer.joblib"]
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "artifacts": {p.name: sha256(p) for p in artifact_paths if p.is_file()},
                "selection_cutoff": str(timestamp(validation_end)),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    # No training or validation review is exported as a downstream forecast feature.
    scored = frame[frame.published_at.gt(timestamp(validation_end))].copy()
    scored["text_score"] = predict(scored.text)
    scored["scorer_cutoff"] = timestamp(validation_end)
    scored["scorer_id"] = f"{backend}-{identity}"
    scored.drop(columns=["scaled_rating", "text_hash", "dedup_group"]).to_csv(
        output / "scored_reviews.csv", index=False
    )
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        part[["review_id", "duplicate_cluster", "published_at", "available_at"]].to_csv(
            output / f"{name}_manifest.csv", index=False
        )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
