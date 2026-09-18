# Canonical input contract

Some historical files have been added and classified in [the data inventory](data-inventory.md). Their upstream processing and exact review/collection timestamps are still missing. The new pipeline deliberately fails on absent inputs and does not infer dates, product mappings, collection completeness, label scales, or Google Trends period semantics from the legacy notebook.

## Target: `trends.csv`

One product/query/geographic target per study. A Sunday-ending Monday–Sunday example:

```csv
date,available_at,value
2020-01-05,2020-01-06T00:00:00Z,45
2020-01-12,2020-01-13T00:00:00Z,51
```

- `date`: 00:00 UTC label on the week’s final calendar day. Use Sunday for Monday–Sunday weeks (`week_end_day: SUN`, `issue_offset_days: 1`) or Saturday for Sunday–Saturday weeks (`week_end_day: SAT`, `issue_offset_days: 2`). The issue time in both cases is the following Monday 00:00 UTC. The label marks the named day, not an observed publication timestamp. Preserve the underlying interval when converting a source export.
- `available_at`: earliest usable publication/receipt timestamp for that observation, in UTC. Never earlier than the end of its weekly period. Labels with delayed reporting cannot enter training until this timestamp. Input target lags obey the same rule.
- `value`: numeric search-interest index in [0,100], or blank for genuinely missing. A source zero remains zero; blank is not zero. Resolve textual suppression markers upstream and document that treatment.
- Duplicate dates are rejected. Absent weeks are reindexed with missing values. Neither targets nor evaluation truth are filled.
- Log exact query/topic, geography, category, search type, retrieval timestamp, and interval conversion in the study config. Partial weeks must be excluded upstream. Store source files and hashes.

Fixed retrospective exports do not recover the values a historical deployment would have observed. The pipeline expressly supports a fixed-snapshot research comparison, not a vintage-aware operational replay. Do not present invented availability as observed history. The supplied export has a [source-interval conversion](../data/prepared/target/trends_source_weeks.csv) and a [Saturday-ending canonical target](../data/prepared/target/trends_ASSUMED_saturday.csv). The latter assumes availability on the Monday after each source Sunday–Saturday week and excludes a likely partial last week based on file modification time. Those assumptions are labeled in every row and [the manifest](../data/prepared/target/trends_assumptions.json). Configure Saturday periods; never relabel their values as Monday–Sunday aggregates. Obtain actual release information or prospective snapshots for operational claims. See [the source-week audit](data-inventory.md).

## Raw reviews: `raw_reviews.csv`

Required columns: `review_id`, `duplicate_cluster`, `product_id`, `published_at`, `available_at`, `rating`, `text`.

- IDs and duplicate clusters must be stable, nonempty strings. Product IDs must come from an audited mapping; no substring matching is performed.
- `duplicate_cluster` should already represent near-duplicate/syndicated reviews. The scorer also joins exact normalized-text matches to those clusters and keeps the earliest available copy of each connected component.
- `published_at` and `available_at` must be exact, absolute timestamps. Availability is no earlier than publication. Approximate “months ago” records need recovery or a separate monthly study.
- `rating` is integer stars 1–5. It may be missing for inference but is required for training/evaluation. Labels are explicitly `(rating - 3)/2`; this is a rating proxy, not human-labeled text sentiment.
- `text` is nonempty review text, with title concatenated upstream if desired. Preserve language/original text and version any translation. The provided BERT base model is English; multilingual validation/translation is not implemented.
- Recommended additional provenance columns: source, source URL/ID, snapshot ID, language, date precision, first-observed timestamp, and product mapping version. They may be retained as additional columns.

The scorer partitions on timestamp cutoffs, uses validation only for selection, evaluates on a later test block, and exports only reviews published after the validation cutoff. Later-arriving reviews published inside the training/validation era are excluded from export; this conservative policy prevents reusing development reviews as purported future features.

## Scored reviews: `scored_reviews.csv`

Required columns: `review_id`, `duplicate_cluster`, `product_id`, `published_at`, `available_at`, `rating`, `text_score`, `scorer_cutoff`, `scorer_id`.

Use the scorer command's output. External scores are accepted only when their provenance satisfies the same contract:

- `text_score` is finite within [-1,1]. A missing score is rejected.
- `scorer_cutoff` is the latest information timestamp used for **training or model selection**, not just the end of gradient training. It strictly precedes every scored review's publication.
- This implementation accepts one frozen scorer/cutoff per selected product. Adaptive or randomly cross-fitted score mixtures are rejected.
- Forecast training origins start after both a 52-week target-history warm-up and a 13-week post-scorer warm-up. Every ablation uses the same training-origin policy.
- Within 4- and 13-week windows, review features describe **newly available reviews by receipt time**. A late historical review contributes when received; it cannot rewrite historical origin features. This choice differs from aggregating by the displayed review date and must be retained in reporting.
- Missing rating/sentiment aggregate values remain missing until fold-local imputation. Review counts and coverage indicators distinguish no observed reviews from observed neutral sentiment.

Code can validate cutoff ordering but cannot prove an externally supplied scorer's claimed lineage. Recover and audit its training/validation manifests before using it as evidence.

## Collection coverage: `coverage.csv`

```csv
date,available_at,complete
2020-01-05,2020-01-06T00:00:00Z,1
2020-01-12,2020-01-13T00:00:00Z,0
```

This file applies to the selected product and audited source set. `complete` is 1 only when scheduled collection for that week was complete; 0 means incomplete/unknown, not zero reviews. `available_at` is when that coverage status was known and cannot precede week completion. Absent coverage rows count as unknown. Do not infer completeness from the presence of reviews.

## Configuration and evaluation

Development/test boundaries are weekly **forecast origins**. All development labels through the maximum horizon must mature before the first test issue time; the code verifies both date boundaries and label availability. The final period uses weekly expanding refits with frozen per-horizon/per-variant ridge penalties. Test outcomes may enter a later scheduled refit after their release, but never hyperparameter selection.

Missing actuals remain unscored and are counted in `metrics.csv`. The forecasting ledger can include unobserved future target dates at the end of the supplied history. Report scored counts with every metric. Review `fits.csv` and the saved pipeline training-origin indices when auditing label eligibility.

Choose new final observations for confirmatory work: the original historical data have already influenced research decisions. Version and register configs before reading evaluation results. The run-directory guard prevents overwrites, but cannot prevent a researcher from repeatedly redesigning a study after looking at results.
