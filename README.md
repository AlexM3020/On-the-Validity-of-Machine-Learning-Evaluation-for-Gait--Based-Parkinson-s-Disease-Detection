# Subject-Independent Evaluation of Mamba for Parkinson's Disease Detection from Gait Signals

The repository now studies **linear-prediction residuals (LPR), compact gait classifiers, and how preprocessing and evaluation choices affect Parkinson's disease detection**. Raw-signal Mamba/CNN experiments are retained as comparators and research history, not the sole scope. The title above is deliberately unchanged pending the author's revision.

## Start here: maintained Kaggle/Colab notebooks

| Notebook | Purpose | Evaluation status |
| :--- | :--- | :--- |
| [LPGNet factorial study](notebooks/LPGNet_Exact_Replication_and_Factorial_Leakage_Study_RestartSafe.ipynb) | Released-design reconstruction and 2×2 LP/validation leakage experiment | Intentionally leaky conditions are explicitly labelled; default original-all cohort retains known duplicate identities |
| [Variant A](notebooks/LPGNet_Variant_A_Fully_Nested_and_Diagnostics.ipynb) | SE/channel-gated CNN with temporal attention pooling | Fully nested participant selection, core-only LP fitting and final fixed-epoch refit; curated cohort |
| [LP-ResMIL](notebooks/LP_ResMIL_Fully_Nested_Evaluation.ipynb) | Multiscale window encoder and masked gated multiple-instance pooling | Fully nested participant selection and refit; padding excluded before encoding; curated cohort |
| [LPR extensions](notebooks/lpr-extensions.ipynb) | Single-order versus multiscale LPR with small statistics-pooling CNNs | Outer CV with inner holdout; no original LPGNet rerun |
| [Raw-signal Mamba/CNN comparison](notebooks/pd-mamba-comparison.ipynb) | Matched compact temporal-backbone comparison | Outer CV with inner holdout; separate experiment, not a residual-model result |

The first three notebooks now use version-pinned [Keras compatibility and checkpoint support](gait/tf_notebook_support.py). The ResMIL masked `tf.where` and remasking Lambdas are replaced by serializable layers with explicit shapes. Real windows are gathered **before** the encoder so padding cannot affect trainable BatchNorm. The mechanism remains **gated attention pooling**, not Q/K/V self-attention. Variant A's pooling is serializable as well.

### Safe execution and recovery

1. Import the desired notebook into a fresh Kaggle/Colab kernel, enable Internet for source/data setup, and select a GPU for long training. Do not combine old cached `gait` imports with a newer notebook.
2. Review cohort policy, seed/search budgets and `CHECKPOINT_DIR` before running. Training is disabled until `RUN_EXPERIMENT=True`. `MAX_NEW_TASKS=1` allows incremental outer-task execution.
3. Run the real model/backpropagation/save-load smoke checks. No unsupported Keras-3 `fit(workers=...)` arguments or unsafe Lambda loading are used.
4. Once training is enabled, **automatic atomic epoch checkpoints** preserve full model/optimizer state, history, LR and early-stopping counters. Inner assessments and outer tasks commit separately. Completed tasks are validated against filenames, participants, labels and probability ranges.
5. Use a new `keras3_v2` run directory. Older CSV-only checkpoints are not silently migrated because model masking/reference fitting changed. The saved plan rejects data, settings, source or runtime changes.
6. For a full runtime reset, explicitly export/download the checkpoint ZIP or use mounted persistent storage. Kaggle working storage and Colab `/content` are **not guaranteed durable across sessions**. An interrupted epoch restarts from the last committed epoch; cross-hardware bitwise equivalence is not promised.

Read [TENSORFLOW_NOTEBOOKS.md](TENSORFLOW_NOTEBOOKS.md) for the exact recovery contract, verified-data policies, modernization differences and limitations. Optional dependencies are in [requirements-tensorflow.txt](requirements-tensorflow.txt); do not replace a working cloud GPU TensorFlow installation unnecessarily.

### Scientific scope and evidence

**Participant independence** remains central, but fitting a reference transform also belongs inside the split. For Variant A and LP-ResMIL, LP fitting excludes both early-stopping and inner-assessment participants. After selecting settings/epochs, refitting on all outer-training participants is allowed; outer test data never drive those choices. Search lists and inner partitions are independent of training seed.

The official dataset contains 306 recordings with 165 recording-backed identities. Exact cross-ID duplicates involve `GaPt17/GaPt18` and `JuCo26/JuPt01`, including conflicting labels. The conservative curated policy excludes all recordings from those four identities and deduplicates within identity, retaining 161 participants and 292 recordings. Original-all factorial runs deliberately retain the original inventory; do not describe their locally fitted arm as certified identity-clean. File checksums and clinical TXT/HTML joins are verified; missing or conflicting HY remains unknown.

Report participant-level OOF results and paired uncertainty as primary, recording metrics separately, and full cohort/search provenance. HY/study summaries, attention weights and high-confidence errors are exploratory diagnostics, not causal findings. Bootstrap intervals condition on fitted predictions; fold SD is not a confidence interval. Uploaded historical notebooks/results are preserved and must be distinguished from measurements of the maintained workflows.

### Validation

The TensorFlow suite exercises actual model construction/backpropagation, padding invariance, safe `.keras` model/optimizer roundtrips, simulated epoch interruption/resume, task-plan integrity, safe bundle restore, and a small nested training task. It was tested with TensorFlow 2.20.0 / Keras 3.15.1 on CPU. This is not a claim that full cohort GPU experiments have been rerun on Kaggle or Colab.

Run the optional TensorFlow checks with `python -m unittest discover -s tests -p test_tf_notebooks.py -v` in the TensorFlow environment. The general suite includes PyTorch tests and needs its separate dependencies. Neither suite installs packages automatically.

---

The sections below document the earlier PyTorch experiments. Their preprocessing, selection strategy and persistence behavior differ from the maintained TensorFlow workflows above.

## New: fold-fitted LPR extensions (no original LPGNet rerun)

[Residual extension notebook](notebooks/lpr-extensions.ipynb) compares **order-11 LPR + mean/std CNN** against **orders 5/11/21 + the same mean/std CNN**. These are new extensions, not reproduction of original LPGNet. Tiny neural models contain 5,446 and 5,518 parameters; each fold fits the control-reference predictors and scaling strictly on inner-training controls. A two-stage window-encoder/participant-head workflow uses matched outer folds, three training seeds, inner-only parameter selection, and optional export/resume. No Mamba CUDA dependency is needed.

Read [LPR_EXTENSION_DESIGN.txt](LPR_EXTENSION_DESIGN.txt) for exact estimator, boundary treatment, training, participant feature aggregation, limitations and the **180 two-stage-fit** default budget. Code: [residual_experiment.py](gait/residual_experiment.py); dependencies: [requirements-lpr.txt](requirements-lpr.txt). No raw data, trained weights or measured new classification results are distributed.

## Current experiment: compact Mamba versus temporal CNN

Use [the Kaggle comparison notebook](notebooks/pd-mamba-comparison.ipynb). It implements the smaller mean/std-pooling Mamba proposal and a dilated temporal CNN, **8 automatic inner-search candidates per model, 10 matched outer participant folds, and 3 training seeds**. Default cost is **480 candidate fits**; chunked execution and explicit export/resume allow multiple sessions. No automatic result files are written.

The notebook verifies all official recordings and clinical metadata, quarantines four ambiguous identities implicated in exact cross-ID duplicates, and preserves a common 161-participant cohort. Candidate windows (500/1,000/2,000), normalization, architecture, pooling, dropout and AdamW settings are selected only on inner validation. Outer results stay hidden until all planned tasks finish. This is outer cross-validation with **inner holdout** search, not full inner K-fold search.

Outputs include full participant errors, study/diagnosis and verified Hoehn–Yahr summaries, confidently wrong cases, inner-validation NLL contributors, across-seed stability, per-model bootstrap intervals and paired model-difference intervals. Clinical information is used only after evaluation, never as training input or a selection criterion. Missing/conflicting metadata remain explicit.

Read [EXPERIMENT_DESIGN.txt](EXPERIMENT_DESIGN.txt) for rationale, ranges, exact preprocessing/validation, compute budgets, output interpretation and limitations. The starting configurations are hypotheses, not claims of improved accuracy. No new trained results or weights are included.

Implementation: [comparison.py](gait/comparison.py), [comparison_models.py](gait/comparison_models.py), [official_data.py](gait/official_data.py), [diagnostics.py](gait/diagnostics.py). Regression tests include [test_comparison.py](tests/test_comparison.py).

## Previous baseline (source and documentation retained for provenance)

The earlier width-128 Mamba notebook was removed on GitHub and remains recoverable in Git history. The following baseline documentation and source remain for provenance. They do **not** include the new comparison or integrated official-data quarantine; use the current comparison notebook above for new runs. The repository contains no raw dataset or pretrained weights.

## Submission evaluation workflow

1. Audit original recordings before splitting: exact numeric-force duplicates within one participant are excluded and logged; duplicates attributed to different participants fail closed. Prepared/chunk-suffixed inputs are rejected. Exact hashes cannot identify all transformed or near-duplicate inputs: use the original PhysioNet release.
2. Construct **10 stratified participant-level outer test folds**. Every eligible participant is tested exactly once per training seed.
3. Within each outer development partition, reserve 20% of its participants for **inner validation**. This is outer cross-validation with an inner holdout, not a full inner cross-validation search.
4. Fit normalization and inverse-window-frequency class weights on **inner training only**. No oversampling, duplicated validation windows, or augmentation is used.
5. Train three prespecified optimizer candidates. Scheduler, early stopping, checkpoint selection, and candidate selection all minimize **inner participant log loss**. Ties retain the first candidate. The model is not refitted on the outer development partition: the selected inner checkpoint is tested directly.
6. Score the untouched outer fold once, after selection. Outer scores are not printed during selection. Finish all folds before inspecting results or proposing further changes.
7. Average window PD probabilities within each recording, then average recording probabilities equally within each participant. Classify PD at probability >= 0.5; do not tune the threshold on outer predictions.
8. Report pooled out-of-fold participant accuracy, balanced accuracy, macro-F1, sensitivity, specificity, ROC-AUC, and the confusion matrix. Window metrics are secondary. Class-stratified participant-bootstrap 95% percentile intervals condition on fitted predictions and class counts; they do not measure full retraining/split uncertainty. Fold standard deviation is descriptive, not a confidence interval.

### Kaggle use and limited experiments

Upload the notebook, enable a GPU and Internet for initial setup, and keep the existing dataset attached. Adjust `DATA_DIR` if needed. The notebook fetches the implementation at an **immutable Git commit** when not running inside the repository; it does not track a moving branch. Enable dependency installation only if the Mamba backend is absent. Code/configuration, split assignments, hashes, histories and predictions are retained in RAM. Export is explicit and optional; no figures or results are automatically saved.

The default model is the compact unidirectional architecture below, not the older bidirectional/MLP/positional-embedding prototype. The initial search has only three candidates: Adam learning rate `5e-5`, Adam `2.5e-5`, and Adam `5e-5` with mild L2 weight decay `1e-4`. All use batch size 32, at most 60 epochs, early-stopping patience 10 and ReduceLROnPlateau patience 5. Gradients are clipped at norm 1. Mamba's internal initialization is preserved.

Start with training seed `[42]` (30 candidate fits). The confirmatory option is `[42, 123, 2026]` on the **same outer and inner assignments** (90 fits). Each seed repeats inner selection and receives its own OOF metrics and intervals; repeated participants are never counted as independent extra observations. Across-seed variation measures training randomness, not variation over new fold assignments. For a smaller fixed-budget baseline use the first candidate only, decided before inspecting outer scores. Runtime depends on Kaggle hardware; the notebook reports measured fit times and provides a real width-32 forward/backward smoke check before training.

A small future context test can compare 1,000 and 2,000 timesteps with the same architecture. Put such choices inside a predeclared inner search or evaluate them as a transparently declared follow-up; do not repeatedly optimize against the outer scores. The current candidate search deliberately changes only optimizer settings.

## Method

**18 gait channels → linear projection → layer normalization → two residual Mamba blocks → masked temporal max pooling → binary classification head.**

Default settings: sequence length 2,000, model width 128, two blocks, state size 32, convolution width 32, and expansion factor 2. Each residual block is followed by layer normalization; pooling excludes trailing padding.

### Why Mamba for gait?

Gait signals contain both **local force patterns** and **context across successive walking cycles**. Mamba combines a local depthwise convolution with a selective state-space update, making it a useful design to investigate both scales within a window. Input-dependent selection can modulate which information is retained, while state size and internal expansion provide configurable representational capacity. These are architectural motivations—not guarantees of superior expressivity, accuracy, or clinical validity.

### Participant-independent inputs

1. Discover full study-qualified participant IDs, such as `GaCo01`, `JuCo01`, and `SiPt02`.
2. Assign participants to disjoint inner training, inner validation and outer testing groups; deterministic window indexing does not change membership.
3. Fit each channel's mean and standard deviation over original inner-training recordings, counting each timestep once even if overlapping windows are requested. Training-recording tails are included in these statistics. Reuse these statistics unchanged for inner validation and outer testing; constant channels use standard deviation 1.
4. Use nonoverlapping 2,000-timestep (20-second) windows by default. Drop incomplete tails and explicitly exclude recordings shorter than one window. Optional overlap never crosses recording boundaries; no padding is used by this workflow.
5. Retain participant membership, original-file/force hashes, exclusions and window boundaries in the in-memory result bundle. Normalization statistics are stored per fold/seed.

`Co` denotes control (class 0); `Pt` denotes Parkinson's disease (class 1). Participant metrics are primary: window support and class proportions must not be described as participant counts. Training loss remains window-weighted, so participants with longer/more recordings have more influence; participant-balanced training is a separate possible experiment.

## Results reported in the paper

| Model | Accuracy | Macro-F1 | Scored windows |
| :--- | ---: | ---: | ---: |
| Mamba | 88.61% | 88.52% | 790 |

Reported recall is **80.20% for Parkinson's disease** and **96.97% for controls**. These are historical development-validation results, not measurements from the corrected notebook or an untouched test set. Their nearly balanced window support must not be interpreted as natural participant prevalence. The corrected protocol must be run to obtain new results; it does not retroactively certify the historical experiment.

## Installation

Use Python 3.10+ in a virtual environment. Shared dependencies are NumPy, PyTorch, scikit-learn, and Matplotlib:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Real Mamba training/evaluation requires **Linux or WSL, an NVIDIA GPU, CUDA-enabled PyTorch, and a compatible CUDA build toolchain**. Ensure PyTorch is the appropriate CUDA-enabled distribution before installing the additional backend:

```bash
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

The **width-32 convolution uses PyTorch depthwise convolution and the real Mamba selective-scan CUDA operation**. It bypasses the small-width optimized convolution path; the optional `causal-conv1d` extension is not required. There is no substitute CPU Mamba backend. The notebook's real forward/backward check must pass on the selected GPU before training.

## Dataset

Obtain the [Gait in Parkinson's Disease dataset from PhysioNet](https://physionet.org/content/gaitpdb/1.0.0/) under its applicable terms. Place signal text files in a local directory; recursive discovery is supported, downloading is not automatic.

Accepted originals have names such as `GaCo01_01.txt`, with study `Ga`, `Ju`, or `Si`, group `Co` or `Pt`, and two-digit participant/walk identifiers. Common copy-name suffixes are audited. Non-data text files are ignored. Original time axes must be increasing and approximately 100 Hz; invalid inputs fail rather than being silently imputed.

Input is a whitespace-separated matrix with **time in column 0 and 18 force channels in columns 1–18** (zero-based). Malformed matrices and nonfinite values raise errors.

Use original recordings or prepared inputs that preserve participant lineage. This workflow performs no oversampling or augmentation; external preparation must not mix participants across partitions.

## Repository cleanup

The single-split training/checkpoint CLI and its runner-specific tests were retired so the submission has one primary evaluation workflow. Older source remains recoverable in Git history. The tested standalone data utilities are retained for compatibility but are **not used by the corrected notebook**; their recording-wise normalization and short-record padding are not this experiment's preprocessing.

The downloaded prototype's unsupported baseline-ranking prose, bidirectional architecture, ineffective augmentation flag, and repeated-last-value learning-curve padding are not carried into the new notebook. No raw evidence, model-contract tests, or historical Git commits were deleted.

## Repository layout

| File | Purpose |
| :--- | :--- |
| [notebooks/pd-mamba-comparison.ipynb](notebooks/pd-mamba-comparison.ipynb) | Current Kaggle Mamba/CNN comparison, audit, training and diagnostics |
| [gait/cross_validation.py](gait/cross_validation.py) | Corrected data audit, nested selection, outer tests, aggregation, CIs and opt-in export |
| [gait/data.py](gait/data.py) | Legacy standalone utilities, not the notebook pipeline |
| [gait/models.py](gait/models.py) | Mamba classifier and width-32 real-scan helper |
| [requirements.txt](requirements.txt) | Shared dependencies |
| [requirements-mamba.txt](requirements-mamba.txt) | Additional CUDA Mamba dependency |
| [tests/test_data_models.py](tests/test_data_models.py) | Data tests, synthetic Mamba contracts, optional real CUDA smoke test |
| [tests/test_cross_validation.py](tests/test_cross_validation.py) | Synthetic leakage/selection, duplicate, weighting, OOF coverage and interval tests |

## Tests

```bash
python -m unittest discover -s tests -v
```

Data tests cover duplicate exclusion, cross-participant duplicate rejection, stratification, train-only normalization/weights and tail exclusion. A complete synthetic multi-fold, two-seed experiment verifies inner-only selection and OOF coverage. An outer-test perturbation test verifies that changing test signals leaves inner checkpoints unchanged. CPU architecture/scan-contract tests use explicitly labeled doubles: they neither execute real Mamba nor produce research results.

An optional test executes real width-32 Mamba forward/backward passes and checks finite outputs/gradients and padding invariance. It explicitly skips when CUDA or the Mamba backend is unavailable. No dependencies are installed by the suite, and a skip is not evidence of CUDA correctness.

## Scope

Research software, not a diagnostic system. Participant separation prevents one important form of leakage but does not establish cross-cohort robustness, clinical validity, or independence of externally prepared data. Seeds and manifests aid repeatability; numerical identity across hardware/CUDA versions is not guaranteed.
