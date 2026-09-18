import argparse
from pathlib import Path
from .backtest import run_from_paths
from .synthetic import write_fixture


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe retrospective search-interest study"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser(
        "synthetic", help="Generate labeled synthetic fixtures, not research results"
    )
    synthetic.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for name in ["config", "trends", "reviews", "coverage", "output"]:
        run.add_argument(f"--{name}", required=True)
    score = sub.add_parser(
        "score-reviews",
        help="Temporal rating-proxy training and frozen future-review scoring",
    )
    for name in ["reviews", "train-end", "validation-end", "test-end", "output"]:
        score.add_argument(f"--{name}", required=True)
    score.add_argument("--backend", choices=["tfidf", "bert"], default="tfidf")
    score.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "synthetic":
            path = write_fixture(args.output)
            print(
                f"Synthetic fixture written to {path.parent}; no real-data conclusions may be drawn."
            )
        elif args.command == "score-reviews":
            from .sentiment import score_reviews

            score_reviews(
                args.reviews,
                args.train_end,
                args.validation_end,
                args.test_end,
                args.output,
                args.backend,
                args.seed,
            )
            print(f"Frozen scorer and temporal evaluation written to {args.output}")
        else:
            ledger = run_from_paths(
                args.config, args.trends, args.reviews, args.coverage, args.output
            )
            print(
                f"Wrote {len(ledger)} forecasts to {Path(args.output).resolve()}; inspect manifest.json for scope and limitations."
            )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Input/study error: {exc}\n")


if __name__ == "__main__":
    main()
