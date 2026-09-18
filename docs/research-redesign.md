# Research validity audit and redesign

Status: audit and proposed design. The initial repair is implemented; see [implementation status](implementation-status.md) for completed work and remaining empirical requirements. Audit evidence and cell references below refer to the archived original notebooks.

## 1. Decision and research question

Keep the idea of testing review information as a forecasting signal, but rebuild the evaluation before extending the neural models. The current repository does not establish that sentiment improves out-of-sample forecasts, and it does not test sales forecasting: the implemented response variable is Google Trends search interest. The observed improvement could reflect leakage, architecture differences, or sampling variation.

Proposed primary question:

> Do review-derived signals available at a forecast origin improve 1-, 4-, and 13-week forecasts of weekly Levi’s 505 search interest beyond historical search interest, calendar seasonality, review volume, and observed star ratings?

Make 4-week-ahead MAE the primary endpoint. Treat 1- and 13-week results as secondary and 52-week forecasts as exploratory until there is enough history or enough comparable series. These are proposed choices to freeze before rerunning experiments, not findings about the best horizon.

A sales study is a separate extension requiring actual weekly sales, a defined product/channel/geographic scope, and price, promotion, returns, and availability information. Search interest is not a validated replacement for either sales or unconstrained demand. Promotion optimization would additionally require an intervention or defensible causal design.

## 2. Scope and evidence limitations

Reviewed all source cells and saved textual outputs in [the forecasting notebook](../notebooks/archive/4_27_direct_prediction_LSTM.ipynb), [the sentiment notebook](../notebooks/archive/BERT_sentiment.ipynb), and [the original README](../notebooks/archive/original-README.md). Cell numbers below are **one-based positions including markdown cells**, not execution counters. The forecasting notebook has one source cell; its numbered section comments locate findings.

The repository contains two notebooks, a README, and seven PNGs. It does not contain `505_trend.csv`, `data_sentiment_merged_v3.csv`, `df_reviews.pkl`, the saved BERT checkpoint, the scraping/linkage/translation scripts, or dependency locks. The transformation connecting BERT review predictions to the four forecasting sentiment columns is absent. Actual data coverage, duplicates, class balance, date alignment, feature provenance, and metric reproducibility therefore remain unverified.

This is a static audit plus a synthetic check of window indexing and architecture parameter counts. Training was not rerun. Saved notebook outputs are historical evidence, not independently reproduced results; the BERT execution counters are non-monotonic and some populated cells have null counters.

## 3. Prioritized findings

P0 blocks a defensible performance claim. P1 is required for a credible study. P2 improves reliability or reporting. “Confirmed” refers to visible code, not a measured amount of bias.

| Priority / status | Evidence | Consequence and required fix |
|---|---|---|
| P0 — confirmed | Forecast §2 fits both `MinMaxScaler` objects on all rows before splitting. | Held-out extrema affect training transformations. Fit preprocessing separately on each training fold and reuse it unchanged on validation/test. |
| P0 — confirmed | `create_multistep_sequences` builds 52-week targets before splitting adjacent windows at `int(0.90 * len(X_num_seq))`. | Training target dates extend beyond the first test origin. Split by label availability and forecast origin, not window index alone; see §4. |
| P0 — confirmed | Forecast §§3–4 pass `X*_te, y_te` to `validation_data`; callbacks monitor `val_loss`; §6 scores the same arrays. | The reported test set selects learning rates and stopping epochs. Add inner temporal validation and a locked outer evaluation. |
| P0 — confirmed unsafe operation; affected rows unknown | Forecast §1 performs an outer merge followed by `.ffill().bfill()` on all columns. | Backfill can import future sentiment or targets; filling target gaps invents ground truth. Remove backward filling and preserve missing targets. Audit actual affected rows. |
| P0 — confirmed | Forecast response is `num_search` from `505_trend.csv`; README claims sales/demand and consistent improvements. | The conclusion exceeds the measured target and evidence. Reframe as exploratory search-interest forecasting. |
| P1 — confirmed confound | Base: one 64-unit LSTM; augmented: 64- then 32-unit LSTMs with different dropout. | Feature value and architecture change together. Use matched models and equal search budgets for every ablation. |
| P1 — confirmed provenance gap | Four sentiment columns arrive already computed; no generating code is present. | Cannot verify trailing versus centered windows, weighting, dates, or cross-fitting. Rebuild features from timestamped review records. |
| P1 — confirmed reuse; downstream contamination conditional | BERT cells 8 and 32–38 randomly split older reviews, then score/export the entire older subset, including training rows. | In-sample scores can behave differently from unseen-review scores. If these exports feed the forecast, the feature generator can also use labels from later historical dates. Use a frozen pre-period scorer or chronological cross-fitting. Recover lineage before asserting this occurred downstream. |
| P1 — confirmed split limitation; duplicates unverified | BERT cell 8 uses two row-wise random splits; no duplicate or source grouping is shown. | This evaluates same-pool rating prediction, not temporal or platform generalization. Audit duplicates and use chronological, group-aware splits. Random splitting alone does not prove duplicate leakage. |
| P1 — confirmed target limitation | BERT cell 8 sets `LABEL_COLUMN = 'scaled_rating'`; original scaling is absent. | Metrics measure prediction of a rating proxy, not independently labeled sentiment. Restore the mapping and compare with ratings directly; add human text annotations if claiming sentiment validity. |
| P1 — conditional | Review dates are represented as “months ago”; forecast uses Sunday-ending resampling and exact-date merging. | Relative dates may not support weekly precision; week-start versus week-end conventions may misalign data. Preserve timestamp precision and define period boundaries explicitly. |
| P1 — confirmed | No naive/seasonal/statistical baselines, repeated seeds, per-horizon metrics, or uncertainty estimates appear. | A single neural comparison cannot establish robust incremental value or practical usefulness. Add the experiment matrix in §8. |
| P2 — confirmed | Forecast split comment says 80/20, code uses 90/10; “full-data forecast” uses models fitted only on `*_tr`. | Documentation is misleading. Save actual date manifests and explicitly refit after final evaluation. |
| P2 — confirmed | BERT cells 14–17 average batch losses equally; test predictions are then clipped before metric calculation. | Partial batches receive disproportionate weight, and logged test loss differs from clipped MSE. Compute sample-weighted metrics and standardize the inference policy. |
| P2 — confirmed | No environment lock, systematic seeds, run manifests, or executable data preparation is committed. | Results cannot be reproduced from the repository. Implement the artifact contract in §10. |

The saved forecast RMSEs are 12.8994 for the base LSTM and 12.4750 for the augmented LSTM: a 0.4244-point, approximately 3.29% reduction. These are not valid estimates of untouched-test improvement under the present protocol.

## 4. Why the multi-step split leaks

Let a window start at index `i`, use lookback `L = 16`, and predict `H = 52` weeks:

- Inputs: `i` through `i + 15`.
- Forecast origin: `i + 15` (assuming the latest input is already published).
- Labels: `i + 16` through `i + 67`.

If the first test window starts at `s`, the final training window starts at `s - 1`. Its labels are `s + 15` through `s + 66`; the first test labels are `s + 16` through `s + 67`. **They share 51 target weeks.** The last training label lies 51 weeks after the first test origin. Many later test targets are also used during fitting, although their positions in output vectors differ.

An illustrative 260-row weekly series yields 193 windows, 173 training windows, and 20 test windows under the current code. Only 122 of those 173 training windows have their complete 52-week label vector available at the first test origin. These counts are a synthetic example, not a claim about the missing CSV.

Correct rule for fitting at origin `t`:

```text
feature_record.available_at <= its simulated prediction origin
training_example.label_end <= t
training_example.label_available_at <= issuance_time(t)
```

For a direct model specific to horizon `h`, require its individual label to be available; it need not wait for all 52 labels. For a shared 52-output model, require the entire target vector unless a carefully designed masked loss is used.

Overlap among past input windows is expected and not itself leakage. Reusing a previously observed outcome in a later, scheduled rolling refit is also legitimate. The violation is using outcomes that had not arrived when a forecast was supposed to be issued. A generic lookback-sized gap does not fix this; eligibility must depend on target endpoints and reporting delays.

## 5. Data and availability contract

### 5.1 Canonical records

Preserve immutable source snapshots and build validated tables:

| Table | Required fields |
|---|---|
| Reviews | stable review ID, duplicate-cluster ID, product ID, source, original text, language, rating, event/publication time, first-observed time, date precision, scrape snapshot ID |
| Product mapping | source product ID, canonical product family/SKU, mapping evidence/confidence, valid dates, mapping version |
| Trends | query/topic, geography, category, search type, period start/end, retrieval time, value, partial-period flag, source snapshot ID |
| Derived review scores | review ID, scorer checkpoint/version, scorer training cutoff, score, processing version, scoring time |
| Weekly features | product/region/week key, feature values, counts, coverage indicators, maximum underlying availability time, feature version |
| Forecast ledger | run/model/fold/seed, issue time, origin, horizon, target date, prediction, interval bounds, eventual actual and actual vintage |

Define the unit initially as Levi’s 505 product-family search interest in a documented geography. Replace substring matching (`contains('505')`) with a reviewed mapping; that filter does not establish correct product identity. Distinguish SKU-level reviews from a family-level target and document weighting.

For operational replay, a record is usable only after it was both published and accessible to the collection process. Old reviews discovered in a later scrape cannot automatically be treated as known at their displayed date. If only retrospective snapshots exist, describe a retrospective historical-data study with assumed availability, not a real-time deployment backtest. Use conservative delay sensitivity analyses and collect prospective snapshots.

### 5.2 Calendar and missingness

1. Define a timezone, week boundaries, and forecast issue time; use only complete, released periods.
2. Parse Trends dates according to export semantics and map both sources to the same period intervals. Do not simply assume a Sunday date denotes a Sunday-ending week.
3. Establish one unique weekly target grid. Validate sorted dates, frequency, duplicates, gaps, and numeric parsing, including suppressed or nonnumeric export values.
4. Left-join review features onto that grid. An outer join can create extra dates and make a “16-week” window span something else.
5. Never fill evaluation labels. Exclude unavailable labels from scoring with documented coverage; for training, drop affected target vectors or use an explicit missing-observation model.
6. Keep missing sentiment distinguishable from neutral sentiment. Include review count, source coverage, and time since last review. If forward carry is tested, cap its age and tune the policy on training/validation only.
7. Distinguish zero reviews under complete collection from collection failure. Deduplicate syndicated reviews before aggregation.

Rounded “months ago” timestamps may only support monthly analysis. Recover exact timestamps where possible; otherwise use a monthly study or sensitivity analysis over plausible date intervals. Do not assign false weekly precision. Monthly aggregation further reduces effective sample size and makes complex models less attractive.

### 5.3 Trends normalization and vintages

Google Trends is sampled relative interest, normalized for the query’s time/location scope and scaled to 0–100; low search volume can be represented as zero. It is not absolute search counts. See [Google’s data FAQ](https://support.google.com/trends/answer/4365533?hl=en).

Record complete request metadata and immutable downloads. A full-history download may encode information about the later normalization window; fixing `MinMaxScaler` does not restore historical data vintages. This is an additional deployment-validity risk, not proof that every full-history normalized series necessarily inflates every metric. If comparing vintages, define a consistent target scale using an audited overlap/anchor procedure with calibration restricted to information available at the origin. Never compare errors across independently rescaled windows as though their units were identical. If vintages cannot be recovered, explicitly label the target as the fixed retrospective export and collect prospective data for operational claims.

## 6. Rebuild the sentiment measurement

### 6.1 What the existing BERT result means

The visible labels are scaled star ratings, not independent sentiment judgments. The notebook reports MAE 0.1328, MSE 0.0787, RMSE 0.2805, and R² 0.8286 after clipping predictions to `[-1, 1]`. The rating-to-score transformation must be recovered before interpreting these units. Do not infer it from the clipping bounds or the unused `label_mapping` dictionary.

Across the saved five epochs, training loss declines from 0.0474 to 0.0157, while validation loss changes from 0.0766 to 0.0748 with fluctuations. This is a sizable generalization gap and a reason to constrain capacity; it does not establish that later epochs monotonically overfit. Validation is actually lowest at the final saved epoch. Add best-checkpoint selection, fixed seeds, and explicit regularization, then assess on an untouched temporal test.

The 16,458 reviews within the latest year are set aside but never evaluated in the visible notebook. This is an opportunity for temporal testing, subject to provenance and duplicate checks, not proof of an existing temporal holdout result. The older subset contains 64,919 reviews in saved output; reconcile these counts and collection snapshots with README volume claims.

BERT is a bidirectional Transformer **encoder**, not an encoder/decoder setup as the README states. Its access to both sides of an already available review is not future time-series leakage. See the [original BERT paper](https://aclanthology.org/N19-1423/).

### 6.2 Proposed scorer evaluation

- Restore the label definition, missing-value rules, and translation/emoji processing. Normalize null title/comment values safely; measure empty-text and 128-token truncation rates. Preserve original text and avoid destroying negation or context with simplistic emoji substitutions.
- Identify exact and near-duplicate review clusters before assignment. Keep clusters from crossing evaluation boundaries, including syndicated reviews. Use stable reviewer identity where available; generic usernames such as “Anonymous” do not identify one person.
- Use older data for training, a later block for validation, and a final later block for testing. Report source/product holdouts separately as transfer tests rather than mixing them with within-product forecasting.
- Compare a constant predictor, TF–IDF plus regularized regression, frozen text embeddings plus a small head, and fine-tuned BERT. Fit vocabularies, learned transforms, and calibration only on allowed training data.
- Treat continuous rating regression as a reasonable baseline, not categorically a modeling error. Compare ordinal rating prediction if the objective is predicting ordered stars. Apply a fixed bounded-output or clipping policy identically in validation and deployment.
- Report per-rating, source, language, time, and text-length errors plus sample counts; include rating distribution and majority/mean baselines. Overall R² does not demonstrate robust text sentiment measurement.
- If the scientific construct is sentiment, annotate a stratified text sample without showing ratings to annotators, use at least two annotators and adjudication, and report agreement. Include fit, durability, sizing, and delivery aspects where useful. Keep this evaluation set separate from development.

### 6.3 Safe forecasting features

Preferred first experiment: train/select a scorer entirely before the forecasting evaluation era, freeze it, and score subsequent reviews. Use a warm-up period and exclude scorer-training reviews from downstream training features where feasible. This is simple and makes the temporal contract auditable.

If adapting the scorer over time is necessary, use **chronological cross-fitting**: train on reviews available before a block, score that next block, repeat, and store the checkpoint cutoff per score. Reserve a pre-period for scorer hyperparameter selection; global future validation of the scorer would reintroduce leakage. Ordinary random out-of-fold prediction removes self-fitting but does not enforce historical availability. Reuse the same scorer-update policy in training and deployment.

Recompute weekly review count, mean rating, negative-rating share, mean text score, and score dispersion from eligible unique reviews. Initially use a small set of trailing 4- and 13-week summaries; define windows in elapsed weeks rather than ambiguous “last 8/15/30 sentiments.” Do not use centered windows. At issue time after week `t`, aggregate only records already available; a blanket shift is not a substitute for availability checks.

For sparse periods, test shrinkage such as `s_t = (n_t * mean_t + k * prior_mean) / (n_t + k)`, with prior and strength learned only from training data. Retain count/missingness indicators. Audit whether current `avg_*` fields are already product averages; averaging averages without counts can change the intended estimand.

## 7. Forecast protocol

Use an expanding-window development backtest and a chronologically later locked evaluation period. Predeclare origins, horizons, primary metric, tuning budgets, feature variants, and refit cadence before seeing new results. Rolling-origin evaluation permits training only on prior observations; multi-step errors should reflect the intended forecast horizon. See [Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html).

For an approximately five-year weekly series, a provisional design is first three years for initial training, fourth year for development origins, and fifth year for locked evaluation. Exact dates depend on recovered coverage and scorer warm-up. If the first three years cannot supply enough valid supervised examples, shorten horizons or expand data; do not silently relax the availability rules.

At each development origin:

1. Construct the as-of snapshot and causal review features.
2. Build eligible training examples whose labels have arrived.
3. Use earlier rolling origins inside that history for hyperparameters and early stopping. Apply the same label-end restrictions at every inner cutoff.
4. Fit preprocessing on the training partition only; transform later inputs without refitting. Values outside a training MinMax range are possible and need not be clipped.
5. Fit the selected model and write forecasts to the ledger before attaching outcomes.

Fit-only-on-training preprocessing follows the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

After development, freeze model families, parameters, transformations, epoch policy, and updating rules. During final evaluation, allow only the predeclared refit schedule, using outcomes that have since arrived. Do not retune on accumulating final results. A final year with rolling weekly origins gives only about 40 origins with complete 13-week outcomes, and only one with a complete 52-week outcome; it does not provide dozens of independent one-year trials.

Because existing results have already been examined, a newly coded split of the same history cannot erase prior researcher exposure. Label it a reconstructed retrospective evaluation. Reserve newly collected future observations for the strongest confirmatory claim.

For future operational forecasts, refit the frozen pipeline on all eligible observed data after evaluation and version the refitted artifacts separately. The current “full-data” plot does not perform that refit.

## 8. Methods and controlled experiments

### 8.1 Model ladder

| Order | Method | Purpose and constraints |
|---|---|---|
| 1 | Last observation, seasonal naive, drift | Establish whether any learned model beats cheap reference forecasts. Use a documented weekly annual alignment; report 52/53-week holiday effects. |
| 2 | ETS and harmonic regression with short ARIMA errors | Represent level/trend and annual calendar structure parsimoniously. Limit seasonal complexity given few annual cycles. |
| 3 — recommended primary learned model | Direct horizon-specific ridge regression | Predict `y[t+h]` from recent target lags, annual lags where available, calendar terms at `t+h`, and review features known at `t`. Interpretable and regularized for small samples. |
| 4 | Shallow gradient-boosted trees on the same lag table | Test modest nonlinear effects with constrained depth, minimum leaf sizes, and equal tuning budgets. |
| 5 | Small matched LSTM | Retain as a secondary replication after the evaluation is repaired; start with one small recurrent layer and compare identical architectures. |
| Later | Pooled/hierarchical product models, aspect sentiment, probabilistic models | Consider only after recovering multiple comparable series and establishing the simpler study. Validate across time and held-out products. |

Harmonic regression represents longer seasonality through Fourier terms and can model residual short-term dependence; see [the method description](https://otexts.robjhyndman.com/fpp3/dhr.html). For a regression-with-ARIMA-errors variant, future exogenous values must be known calendar inputs, explicitly forecast, or given as labeled scenarios. Never supply realized future reviews, promotions, or weather. Direct models using origin-time features avoid needing future sentiment values.

The existing architectures have approximately 20,276 and 32,052 trainable parameters respectively, calculated from their standard Keras LSTM and Dense configurations. With roughly five years of data, highly overlapping weekly windows provide far fewer independent observations than the review count suggests. Sixty thousand reviews do not create sixty thousand independent search-interest targets. A 16-week recurrent input also lacks explicit annual history; sentiment should not be credited with “replacing seasonality” before testing seasonal predictors.

### 8.2 Feature ablations

Run the following on identical origins, target availability, model class, training budget, and seeds:

| Variant | Inputs | Question |
|---|---|---|
| A | Target lags + calendar | How strong is a well-specified time-series baseline? |
| B | A + review volume and collection coverage | Is apparent sentiment value simply review activity? |
| C | B + observed ratings | Does cheap rating information explain the gain? |
| D | B + text-derived scores | Can text substitute for ratings? |
| E | C + text-derived scores | Does text add information beyond ratings? Primary sentiment comparison is E versus C. |
| F — later | E + selected aspect scores | Do specific complaints help beyond global polarity? |

For neural comparisons, keep layer widths, dropout, optimizer, epoch selection, and tuning budgets fixed; input dimension still changes parameter count slightly, so add constant-feature controls or capacity-matched checks. Also compare both old architectures with and without sentiment to separate the original confound.

Use at least five predetermined seeds for stochastic final candidates and report all results, not the best seed. A proposed small budget is at most 12 configurations per learned model family during development; keep an experiment registry to expose researcher degrees of freedom.

Robustness checks: review publication delays of 0/1/2/4 weeks; exact-date versus uncertain-date subsets; source exclusion; alternative aggregation and deduplication rules; seasonality on/off; and block-permuted review features that preserve some serial structure. Shuffling is a negative control, not a causal test or a replacement for a leakage audit. Choose these analyses before the final evaluation and report the full set.

## 9. Evaluation, uncertainty, and claims

Store one row per origin/horizon/model/seed. Report MAE and RMSE in the target’s documented units at each horizon, with a separately declared aggregate. For an aggregate RMSE, take the square root after averaging squared errors; do not silently average horizon RMSEs. The original multioutput MSE is not inherently a mathematical error, but it hides horizon-specific behavior and repeatedly weights dates appearing in multiple forecast vectors.

Use seasonal MASE as a secondary cross-series measure with its denominator calculated on the fold’s training history; flag undefined/near-zero denominators. Avoid MAPE as the primary metric because Trends can contain zeros. Display errors over origins, seasonal peaks, review coverage, and product/source slices, including counts and missing-label coverage. The first test forecast plot is illustrative, not a complete evaluation.

For paired comparisons, compute loss differences on the same origins and horizons. Estimate uncertainty using paired contiguous-block resampling of origin-level losses, retaining all horizons/seeds together for each sampled origin. Prespecify block-length sensitivity; long overlapping horizons require long dependence blocks and may leave too little data for credible intervals. Do not treat seeds, shared target weeks, or thousands of origin/horizon cells as independent replications. If independent temporal support is inadequate, report descriptive estimates and collect more history rather than claiming significance.

Add 80% and 95% prediction intervals using a model-based baseline or past out-of-sample residual calibration by horizon. Calibration must use development or earlier available residuals only. Report coverage and width; do not assert distribution-free coverage under serial dependence and distribution shift without appropriate assumptions.

Proposed promotion gate for the sentiment method: at least a 5% relative reduction in the primary 4-week MAE against C on development, then confirmation on the locked evaluation with a paired uncertainty interval supporting improvement, no material degradation at secondary horizons, and acceptable stability across periods/seeds. The 5% is a provisional practical threshold, not a statistical fact; freeze an appropriate threshold before experimentation. If the final result is inconclusive, state that. Do not keep changing the threshold or method until the final set passes.

Reviewers are selected customers; review volume, ratings, and search interest can share causes such as promotions, product changes, and advertising. Reviews can lag purchases. Predictive gains would establish incremental predictive association in the tested setting, not that sentiment causes demand or that manipulating sentiment would improve sales.

## 10. Implementation plan and acceptance criteria

### Phase 0 — recover and constrain the evidence

Recover source snapshots, request metadata, preprocessing/linkage code, label mapping, and checkpoint lineage. Produce a data inventory with coverage, date precision, duplicates, missingness, and source counts. Update the README’s conclusions to exploratory language and correct the BERT description. If timestamps support only monthly analysis, explicitly change the study granularity.

**Exit:** another researcher can identify every raw input, target unit, timestamp assumption, and derived feature definition. If the missing assets cannot be recovered, build a new prospective dataset and retain the original notebook only as an exploratory artifact.

### Phase 1 — implement the evaluation core

Suggested structure:

```text
configs/study.yaml           # origins, horizons, metrics, seeds, budgets
src/data/ingest.py           # schema checks and immutable snapshot manifests
src/data/reviews.py          # deduplication and product mapping
src/features/asof.py         # availability-aware aggregation
src/sentiment/scoring.py     # frozen / chronological scorer policy
src/evaluation/splits.py     # origin and label-availability rules
src/evaluation/backtest.py   # nested development and locked evaluation
src/evaluation/metrics.py    # paired, horizon-specific scoring
src/models/baselines.py
src/models/direct.py
src/models/lstm.py
scripts/run_study.py
reports/                    # tables generated from forecast ledger
artifacts/<run_id>/          # configs, hashes, transforms, forecasts, logs
```

Keep notebooks for explanation and plots; move reusable logic into importable modules. Pin Python, framework, tokenizer/model revisions, and dependencies. Save code revision, input hashes, split manifests, random seeds, device information, fitted transformations, and selected hyperparameters for each run. Save tokenizer and preprocessing configuration with the BERT checkpoint. Dataset access can remain restricted while publishing schemas, manifests, and a synthetic fixture.

Required tests target scientific failure modes:

- A synthetic 52-step boundary case rejects the 51 unavailable training windows illustrated above.
- Mutating any data after an origin leaves that origin’s eligible training data, learned transforms, and features unchanged under a fixed seed.
- No selected feature record or training label has an availability timestamp after its permitted cutoff.
- No backward imputation occurs; missing target weeks remain missing and cannot contribute fabricated scoring labels.
- Week alignment handles explicit start/end conventions, duplicates, missing weeks, and year transitions.
- Duplicate review clusters do not cross scorer train/evaluation partitions; scorer training cutoff precedes every chronologically scored block.
- Scalers fit only allowed training rows, and a deliberate future outlier cannot change training parameters.
- Outer evaluation data cannot enter hyperparameter selection or early stopping; any scheduled update uses only labels already released.
- Forecast dates equal origin plus horizon; metrics match hand-calculated examples and sample weighting.
- A clean-environment synthetic run produces a complete ledger and report; the real-data run fails clearly when required inputs are absent.

**Exit:** all temporal invariants pass and naive baselines run end to end before any neural fitting.

### Phase 2 — run the minimum credible study

Implement ridge and seasonal baselines, frozen sentiment scoring, and A–E ablations. Complete development selection with the declared budgets, then run the frozen final evaluation once. Produce per-horizon comparisons, paired uncertainty, seed variability, and data-quality sensitivity results.

**Exit:** a reader can reproduce each table from the forecast ledger and assess whether text improves on ratings and volume. A null or negative result is a valid completed research outcome.

### Phase 3 — extend only when justified

Add the controlled small LSTM, aspect scores, or pooled product models only if the core study is sound and data support the added complexity. Start an actual sales study only with a separately defined sales target. Model stock availability explicitly if interpreting observed sales as demand, and retain future promotions/weather as known-at-origin inputs or scenarios.

## 11. Intended deliverable from the redesigned study

Publish the precise research question, data/provenance card, availability assumptions, preregistered comparison matrix, complete forecast ledger, reproducible environment, and limitations alongside the results. A defensible conclusion should name the target, geography, time period, horizons, baseline, and uncertainty. Until those results exist, replace “sentiment consistently improves forecasting” with “this repository explores whether review-derived signals improve search-interest forecasting; the current evaluation requires correction before performance claims can be made.”
