# Subject-Independent Evaluation of Mamba for Parkinson's Disease Detection from Gait Signals

Mamba-only research code for binary Parkinson's disease classification from multichannel gait signals. **Participant independence** is the central evaluation principle: every recording and window from a participant stays in one partition, including participants with multiple recordings.

The release includes preprocessing, a Mamba classifier, training, checkpoint evaluation, and tests. It contains no manuscript, raw dataset, or pretrained weights.

## Method

**18 gait channels → linear projection → layer normalization → two residual Mamba blocks → masked temporal max pooling → binary classification head.**

Default settings: sequence length 2,000, model width 128, two blocks, state size 32, convolution width 32, and expansion factor 2. Each residual block is followed by layer normalization; pooling excludes trailing padding.

### Why Mamba for gait?

Gait signals contain both **local force patterns** and **context across successive walking cycles**. Mamba combines a local depthwise convolution with a selective state-space update, making it a useful design to investigate both scales within a window. Input-dependent selection can modulate which information is retained, while state size and internal expansion provide configurable representational capacity. These are architectural motivations—not guarantees of superior expressivity, accuracy, or clinical validity.

### Participant-independent inputs

1. Discover full study-qualified participant IDs, such as `GaCo01`, `JuCo01`, and `SiPt02`.
2. Stratify participants into training/validation and, optionally, held-out test partitions **before windowing**.
3. Z-score each recording/channel independently; no normalization statistics are fitted across participants.
4. Create nonoverlapping windows. Discard incomplete tails of long recordings; emit one masked, zero-padded window for a short recording.
5. Save participant membership, recording hashes, and window boundaries for repeatable checkpoint evaluation.

`Co` denotes control (class 0); `Pt` denotes Parkinson's disease (class 1). Metrics are per window: sample support is not a participant count.

## Results reported in the paper

| Model | Accuracy | Macro-F1 | Scored windows |
| :--- | ---: | ---: | ---: |
| Mamba | 88.61% | 88.52% | 790 |

Reported recall is **80.20% for Parkinson's disease** and **96.97% for controls**. These are historical validation results, not new measurements from this release or an untouched test set; validation informed model selection and early stopping. The release reconstructs the documented design, but the exact generating revision and split manifests were not retained, so reproducing these numbers is not guaranteed.

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

The **width-32 convolution uses PyTorch depthwise convolution and the real Mamba selective-scan CUDA operation**. It bypasses the small-width optimized convolution path; the optional `causal-conv1d` extension is not required. There is no substitute CPU Mamba backend. `--device auto` resolves to CUDA or fails; `--device cpu` always raises a clear error before constructing a model or creating run outputs.

## Dataset

Obtain the [Gait in Parkinson's Disease dataset from PhysioNet](https://physionet.org/content/gaitpdb/1.0.0/) under its applicable terms. Place signal text files in a local directory; recursive discovery is supported, downloading is not automatic.

Accepted recording names use study prefixes `Ga`, `Ju`, or `Si`, group `Co` or `Pt`, and a two-digit participant ID; numeric recording and chunk suffixes retain that participant identity. Duplicate basenames are rejected; metadata text files are ignored.

Input is a whitespace-separated matrix with **time in column 0 and 18 force channels in columns 1–18** (zero-based). Use `--channels` only for a different documented layout. Malformed matrices and nonfinite values raise errors.

Use original recordings or prepared inputs that preserve participant lineage. This workflow performs no oversampling or augmentation; external preparation must not mix participants across partitions.

## Training

Run from the repository root with a new, empty output directory:

```bash
python train.py --data-dir data/raw --output-dir outputs/mamba --device cuda
```

Defaults: 80/20 participant training/validation split, seed 42, Adam learning rate `5e-5`, batch size 64, cross-entropy weights `[1.1, 1.0]`, up to 100 epochs, early-stopping patience 10, and learning-rate plateau patience 5.

For a new experiment with approximately 60/20/20 training/validation/test participants:

```bash
python train.py --data-dir data/raw --output-dir outputs/mamba_test --device cuda --validation-fraction 0.2 --test-fraction 0.2
```

Only validation determines early stopping and checkpoint selection; test scoring happens afterward. This option does not describe the historical result above. Inspect all settings with `python train.py --help`.

## Outputs and checkpoint evaluation

Runs save configuration, participant/window manifests, a best-validation state checkpoint, epoch history, a loss curve, and validation metrics/predictions. Test metrics/predictions are included when requested. Checkpoint metadata fixes `model` to `mamba`; there is no model-selection option.

Evaluate a release checkpoint without retraining:

```bash
python train.py --data-dir data/raw --output-dir outputs/evaluation --checkpoint outputs/mamba/best_model.pt --partition validation --device cuda
```

Use `--partition test` only when the checkpoint includes a test partition. Evaluation restores settings and checks filenames, contents, participants, and window boundaries; mismatches are rejected. Only load trusted checkpoints, even with restricted state loading.

## Repository layout

| File | Purpose |
| :--- | :--- |
| [train.py](train.py) | Command-line training and checkpoint evaluation |
| [gait/data.py](gait/data.py) | Recording discovery, participant splits, normalization, and windows |
| [gait/models.py](gait/models.py) | Mamba classifier and width-32 real-scan helper |
| [gait/experiment.py](gait/experiment.py) | Training, early stopping, metrics, plots, and provenance checks |
| [requirements.txt](requirements.txt) | Shared dependencies |
| [requirements-mamba.txt](requirements-mamba.txt) | Additional CUDA Mamba dependency |
| [tests/test_data_models.py](tests/test_data_models.py) | Data tests, synthetic Mamba contracts, optional real CUDA smoke test |
| [tests/test_experiment.py](tests/test_experiment.py) | Synthetic runner tests and production CUDA guards |

## Tests

```bash
python -m unittest discover -s tests -v
```

Data tests cover participant grouping, stratification, normalization, and padding. CPU architecture/scan-contract tests use explicitly labeled doubles. The end-to-end runner test patches device resolution and model construction to use a **synthetic trainable masked-pooling/linear classifier**, checking training, checkpoint prediction parity, and changed-input rejection. It neither executes real Mamba nor produces research results; production CUDA guards are tested separately.

An optional test executes real width-32 Mamba forward/backward passes and checks finite outputs/gradients and padding invariance. It explicitly skips when CUDA or the Mamba backend is unavailable. No dependencies are installed by the suite, and a skip is not evidence of CUDA correctness.

## Scope

Research software, not a diagnostic system. Participant separation prevents one important form of leakage but does not establish cross-cohort robustness, clinical validity, or independence of externally prepared data. Seeds and manifests aid repeatability; numerical identity across hardware/CUDA versions is not guaranteed.
