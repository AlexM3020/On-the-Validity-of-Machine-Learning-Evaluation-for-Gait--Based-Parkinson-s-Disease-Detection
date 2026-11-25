# Beyond Participant-Disjoint Splits: Evaluation Validity for Gait-Based Parkinson's Disease Detection

Research code and retained outputs for the anonymous manuscript **“Beyond Participant-Disjoint Splits: Evaluation Validity for Gait-Based Parkinson's Disease Detection”**, under review at Transactions on Machine Learning Research (TMLR).

This repository studies a methodological question: **which information flows must remain independent of outer-test participants when evaluating gait-based Parkinson's disease (PD) classifiers?** Participant-disjoint folds are necessary but not sufficient. Data-dependent preprocessing and training control must also exclude outer-test information.

The work combines a source-level audit and deterministic partition replay, a controlled LPGNet protocol reconstruction, a single-split LSTM illustration, and a separate participant-independent CNN/Mamba benchmark. It is research software, not a diagnostic system.

## Contributions

- **Three-level evaluation-independence framework.** Partition independence, preprocessing independence, and training-control independence are conjunctive requirements for unseen-participant evaluation.
- **Reproducibility and participant-overlap audit.** Released implementations are examined at pinned revisions; selected recording-level fold constructors are replayed using study-qualified participant IDs and an explicitly chosen audit seed.
- **Controlled LPGNet evaluation study.** Global versus fold-local linear-prediction (LP) fitting is crossed with outer-test versus inner-only training control. Contrasts compare complete protocols rather than isolating a causal effect of test exposure alone.
- **Participant-level reporting and uncertainty.** Window probabilities are averaged within recording and then within participant; pooled participant-level out-of-fold (OOF) metrics are distinguished from recording-level fold means and window-level metrics.
- **Separate direct-model reference.** CNN and Mamba classifiers use matched participant-disjoint outer folds, inner-only selection, three training seeds, and a fixed 0.5 decision threshold. They are not matched leaderboard comparisons with LPGNet.

## Principal retained results

| Experiment | Result | Interpretation |
| :--- | :--- | :--- |
| LSTM development/validation illustration | 92.50% accuracy with recording-level partitioning; 83.08% with participant-disjoint partitioning | Descriptive partition sensitivity; evaluation sets are not participant-matched and these are not independent test estimates |
| LPGNet `global__test` | 86.06% participant-level accuracy; AUC 0.9147 | Released-style reconstruction; global LP and outer-test training control are intentionally exposed |
| LPGNet `local__test` | 84.24% accuracy; AUC 0.9120 | Outer-test training control remains |
| LPGNet `global__inner` | 72.73% accuracy; AUC 0.8154 | Inner training control, but global LP still includes outer-test controls |
| LPGNet `local__inner` | 70.91% accuracy; AUC 0.8359 | Corrects the two tested mechanisms; retains the official inventory's known duplicate and conflicting-label groups |
| Direct CNN | Seed-mean accuracy 75.98%; AUC 0.824 | Separate curated 161-participant benchmark |
| Direct Mamba | Seed-mean accuracy 73.50%; AUC 0.820 | Same outer folds as CNN; no clear architectural advantage is established |

Both test-control versus corresponding inner-control LPGNet accuracy contrasts are **+13.33 percentage points**. These are complete-protocol contrasts: inner arms train on an 85% core-training subset without a final refit, and `local__inner` also estimates its LP reference from fewer controls than `local__test`.

## Repository entry points

| Path | Purpose |
| :--- | :--- |
| [`notebooks/LPGNet_Exact_Replication_and_Factorial_Leakage_Study_RestartSafe.ipynb`](notebooks/LPGNet_Exact_Replication_and_Factorial_Leakage_Study_RestartSafe.ipynb) | Clean, restart-safe LPGNet reconstruction and four-condition factorial runner |
| [`notebooks/pd-mamba-comparison.ipynb`](notebooks/pd-mamba-comparison.ipynb) | Clean direct CNN/Mamba nested-selection runner |
| [`gait/tf_notebook_support.py`](gait/tf_notebook_support.py) | Hash-pinned TensorFlow/Keras factorial implementation and durable checkpoints |
| [`gait/comparison.py`](gait/comparison.py), [`gait/comparison_models.py`](gait/comparison_models.py) | Direct candidate generation, model construction, and resumable execution |
| [`gait/cross_validation.py`](gait/cross_validation.py) | Participant splitting, normalization, aggregation, metrics, and bootstrap utilities |
| [`gait/official_data.py`](gait/official_data.py) | Checksum-verified PhysioNet inventory, curation, and exact clinical joins |
| [`gait/diagnostics.py`](gait/diagnostics.py) | Post-evaluation participant, study, fold, and Hoehn–Yahr diagnostics |
| [`scripts/replay_partition_audit.py`](scripts/replay_partition_audit.py) | Metadata-only deterministic fold replay; no waveform download |
| [`scripts/export_paper_artifacts.py`](scripts/export_paper_artifacts.py) | Exports and checks all 18 manuscript tables and 4 figures |
| [`paper_assets/`](paper_assets/) | Retained LSTM figures/reports and verified factorial loss histories |
| [`DURABLE_RUNS.md`](DURABLE_RUNS.md) | Off-runtime checkpoint setup and recovery contract |
| [`EXPERIMENT_DESIGN.txt`](EXPERIMENT_DESIGN.txt) | Direct benchmark protocol and computational budget |
| [`TENSORFLOW_NOTEBOOKS.md`](TENSORFLOW_NOTEBOOKS.md) | TensorFlow modernization, provenance, and execution contract |

Other notebooks and result directories are retained as historical research evidence. They are not entry points for the current manuscript's reported experiments. Executed notebooks and their archives are kept unchanged as provenance.

## Dataset access

The experiments use the public [PhysioNet Gait in Parkinson's Disease database, version 1.0.0](https://physionet.org/content/gaitpdb/1.0.0/). Obtain and use it under the applicable PhysioNet terms. Raw waveforms, trained weights, and runtime credentials are not distributed here.

`gait/official_data.py` verifies the official `SHA256SUMS.txt` inventory against manifest SHA-256:

```text
44903ce52d5428ba218a9cdfd60d5646ab6674df4926233c756ae7c3c7cd50ec
```

Original recording names retain study-qualified identities such as `GaCo01`, `JuCo01`, and `SiPt02`. The three inventory descriptions answer different questions:

| Inventory | Recordings | Participants | PD / control |
| :--- | ---: | ---: | ---: |
| Raw demographic description | — | 166 | 93 / 73 |
| LPGNet `official_all` inventory | 306 | 165 | 93 / 72 |
| Curated direct-model inventory | 292 | 161 | 90 / 71 |

The direct inventory excludes `GaPt17`, `GaPt18`, `JuCo26`, and `JuPt01` because of cross-identity exact duplicates, including a conflicting-label group. It also counts one byte-identical within-identity repeated copy once. The factorial `official_all` inventory deliberately retains those groups for released-style reconstruction; `local__inner` must therefore not be described as identity-clean.

Clinical TXT/HTML metadata are joined only by exact study-qualified ID. The malformed metadata ID `Juc010` is logged rather than silently repaired. Clinical fields enter only post-evaluation diagnostics, never model fitting, selection, or threshold tuning.

## Environments

### Current compatible dependency ranges

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

For the TensorFlow notebook:

```bash
python -m pip install -r requirements-tensorflow.txt
```

For real Mamba execution, install a compatible CUDA-enabled PyTorch build first, then:

```bash
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

Real Mamba execution requires Linux or WSL, an NVIDIA GPU, CUDA-enabled PyTorch, and a compatible build/runtime environment. The CPU test double is not a substitute Mamba implementation.

### Archived publication-run cores

The immutable result archives record the following execution environments:

| Experiment | Python | Core packages | Hardware/runtime |
| :--- | :--- | :--- | :--- |
| Direct CNN/Mamba | 3.12.13 | NumPy 2.4.6; PyTorch 2.10.0+cu128; scikit-learn 1.6.1; mamba-ssm 2.3.2.post1 | CUDA 12.8; Tesla T4 |
| LPGNet factorial | 3.13.16 | TensorFlow 2.21.0; Keras 3.13.2; NumPy 2.4.6; SciPy 1.16.3; scikit-learn 1.6.1 | Recorded in `run_plan.json` |

The corresponding core pins are supplied in `requirements-direct-archived.txt` and `requirements-factorial-archived.txt`. The CUDA-specific PyTorch build is documented separately in the direct environment file. Numerical identity across different hardware, CUDA versions, or GPU stochastic execution is not claimed.

## Reproduce the paper artifacts without retraining

The exporter reads the tracked result archives and retained assets. It downloads only checksum-verified PhysioNet manifest/clinical metadata into the specified cache; it does not download waveforms or run training.

```bash
python scripts/export_paper_artifacts.py \
  --cache .cache/physionet \
  --output generated_paper
```

The command writes CSV tables, PDF/PNG figures, and `generated_paper/manifest.json`. It checks inventory completeness, participant alignment, all expected cohort counts, the LSTM report values, the external replay counts, and the retained paired contrasts. LPGNet pooled and paired participant intervals use 1,000 class-stratified resamples with seed 2028. Direct archived bootstrap settings are 2,000 resamples with seed 7301.

To run only the deterministic external-partition replay:

```bash
python scripts/replay_partition_audit.py \
  --cache .cache/physionet \
  --output generated_paper/tab-audit-replay.csv
```

These audit partitions are conditional code-path demonstrations with audit seed 42, not reconstructions of the original publications' folds or random states.

## Reproduce the full experiments

### Direct CNN/Mamba benchmark

Open `notebooks/pd-mamba-comparison.ipynb` in a fresh GPU runtime. Training and result export are opt-in (`RUN_EXPERIMENT=False`, `SAVE_RESULTS=False`). After the source, data, and GPU checks, enable training/export explicitly and retain the frozen configuration:

- 10 diagnosis-stratified participant outer folds, split seed 42;
- inner 20% participant holdout;
- 8 candidates per family, search seed 90210;
- training seeds 42, 123, and 2026, with actual task seed `(seed + 1009 * fold) % 2**32`;
- 60 family/fold/seed tasks and 480 candidate fits;
- fixed threshold 0.5 and no final refit on combined core and inner participants.

`MAX_NEW_TASKS` supports chunked execution. Resume only with the same frozen plan and validated result bundle. The selected inner checkpoint is evaluated on the outer fold without refitting.

### LPGNet factorial study

Open `notebooks/LPGNet_Exact_Replication_and_Factorial_Leakage_Study_RestartSafe.ipynb` in a fresh TensorFlow GPU runtime. The clean notebook selects `RUN_PROFILE='full'` and keeps `RUN_EXPERIMENT=False` until the storage and model preflights pass.

The full plan comprises **40 two-stage fits (80 training stages)** across four conditions and ten outer folds. The active configuration is `SEED=1`, `FOLD_SEED=0`; fold `k` uses `SEED+k` for both the 15% inner participant split and training initialization/shuffling. This is one deterministic seed schedule, not ten independent repeated experiments. No candidate-search or separate split seed enters the fixed-architecture factorial control flow.

Use verified off-runtime storage and the same `RUN_ID` to resume. **Configure rclone authentication outside notebook cells; never paste credentials or tokens here.** See [`DURABLE_RUNS.md`](DURABLE_RUNS.md) before starting a long run.

## Tests

```bash
python -m pytest -q
```

The suite covers participant splitting, duplicate handling, train-only normalization, model contracts, candidate generation, result validation, notebook source safety, durable-storage failure handling, and artifact integrity. CUDA/Mamba and TensorFlow tests skip explicitly when their optional runtime is unavailable; a skip is not evidence of hardware-specific correctness.

## Citation

See [`CITATION.cff`](CITATION.cff). The anonymous manuscript citation is:

```bibtex
@misc{anonymous2026evaluationvalidity,
  author = {{Anonymous}},
  title  = {Beyond Participant-Disjoint Splits: Evaluation Validity for Gait-Based Parkinson's Disease Detection},
  year   = {2026},
  note   = {Anonymous manuscript under review at Transactions on Machine Learning Research}
}
```

## License and contact

Licensing terms and personal contact details are intentionally omitted from this anonymous snapshot. PhysioNet data remain subject to their own applicable terms.

## Complete manuscript table and figure map

All paths below are generated under `generated_paper/` by the artifact command above. “Direct archive” refers to `notebooks/pd-mamba-cnn-results/comparison_progress_20261116T204946297832Z.zip`; “factorial archive” refers to `notebooks/results_lpgnet_replication/reports.zip`.

| Manuscript label | Generated file(s) | Source and computation |
| :--- | :--- | :--- |
| `tab:cohort-map` | `tab-cohort-map.csv` | Direct archive audit/OOF inventory, factorial archive participants, and the manuscript's metadata-level demographic description |
| `tab:audit-replay` | `tab-audit-replay.csv` | `replay_partition_audit.py`; pinned PhysioNet filename manifest, audit seed 42, fold zero |
| `tab:expanded-audit` | `tab-expanded-audit.csv` | Manuscript-supported source-inspection findings and reconstruction limits |
| `tab:rnn-split` | `tab-rnn-split.csv` | `paper_assets/lstm/classification_report_recording.csv` and `classification_report_participant.csv` |
| `fig:rnn-leakage-curves` | `fig-rnn-leakage-curves.pdf`, `.png` | Original retained `accuracy_curve_11.png` and `loss_curve_11.png`, composed without altering their plotted data |
| `tab:lpgnet-replication` | `tab-lpgnet-replication.csv` | Factorial archive `fold_metrics.csv`; mean/sample SD over ten `global__test` recording-level folds and published reference values |
| `tab:factorial-protocols` | `tab-factorial-protocols.csv` | Factorial run-plan configuration and manuscript protocol definitions |
| `tab:lpgnet-factorial` | `tab-lpgnet-factorial.csv`, `tab-lpgnet-participant-intervals.csv` | Factorial archive `summary.csv` and `participants.csv`; pooled participant metrics, confusion counts, and seed-2028 bootstrap |
| `tab:factorial-contrasts` | `tab-factorial-contrasts.csv` | Factorial participant OOF predictions; paired class-stratified bootstrap, 1,000 resamples, seed 2028; checked against retained export |
| `fig:lpgnet-roc` | `fig-lpgnet-roc.pdf`, `.png` | Participant-level pooled OOF ROC curves from factorial `participants.csv` |
| `fig:lpgnet-confusion` | `fig-lpgnet-confusion.pdf`, `.png` | Participant-level pooled OOF confusion matrices at threshold 0.5 |
| `tab:direct-results` | `tab-direct-results.csv` | Direct archive `summaries`; means over training seeds 42, 123, and 2026 |
| `tab:direct-hy` | `tab-direct-hy.csv` | Direct OOF errors joined post-evaluation to checksum-verified clinical metadata; seed means/ranges and mean probabilities |
| `tab:direct-study` | `tab-direct-study.csv` | Direct OOF errors by study and diagnosis; seed-mean correctness |
| `tab:lpgnet-provenance` | `tab-lpgnet-provenance.csv` | Manuscript's pinned released/local source-location map |
| `tab:cnn-candidates` | `tab-cnn-candidates.csv` | Direct archive candidate configurations, measured parameter counts/receptive fields, selected counts, and selected-epoch summary |
| `tab:mamba-candidates` | `tab-mamba-candidates.csv` | Direct archive candidate configurations, measured parameter counts, selected counts, and selected-epoch summary |
| `tab:direct-participant` | `tab-direct-participant.csv` | Direct archive seed-specific participant-level metrics and confusion counts |
| `tab:direct-window` | `tab-direct-window.csv` | Direct archive seed-specific secondary window-level metrics |
| `tab:direct-folds` | `tab-direct-folds.csv` | Direct split manifest, verified HY metadata, and seed-mean fold accuracy |
| `tab:persistent-errors` | `tab-persistent-errors.csv` | Participants misclassified by both architectures in all three seeds; normalized one-row-per-participant CSV |
| `fig:lpgnet-loss` | `fig-lpgnet-loss.pdf`, `.png` | Forty authenticated recording-stage histories in `paper_assets/lpgnet/recording_loss_histories.json`; fold traces, medians, and interquartile ranges |
