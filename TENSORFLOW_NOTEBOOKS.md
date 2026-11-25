# TensorFlow notebook compatibility and checkpoint contract

The LPGNet factorial notebook is the TensorFlow entry point for the current
manuscript. Variant A and LP-ResMIL are retained as separate historical research
experiments and are not sources of the manuscript's reported factorial results.

* **LPGNet factorial leakage study:** explicitly contrasts global/local LP fitting
  and test/inner validation. The default `official_all` cohort retains all 306
  original recordings (165 recording-backed IDs), including known ambiguous
  duplicates, for reconstruction. These results are **not clean unseen-participant
  estimates**, even for the locally fitted arm, because identity ambiguity remains.
  Set `COHORT_POLICY='curated'` in a **new run directory** to evaluate all four
  conditions on the same curated cohort. Test-driven conditions are intentional
  leakage experiments and must remain labelled as such.
* **Variant A:** fully nested participant-level configuration selection, a separate
  early-stopping subset, then fixed-epoch refitting on all outer-training people.
  It runs only Variant A. LP is fitted on the early-stopping **core**, not early
  or inner-assessment controls. Search lists and inner splits are now independent
  of training seed. Default cohort: curated 161 participants/292 recordings.
* **LP-ResMIL:** the same fully nested separation and final refit, with
  participant-balanced window/recording losses and gated window-attention pooling.
  It does not retrain reference models. Default cohort: curated.

## Keras changes

The masked `tf.where`, remasking and weighted-sum Lambdas were replaced with
registered serializable layers having explicit output-shape contracts. This is
**gated MIL pooling**, not a Q/K/V self-attention replacement. The masked softmax
uses float32, zeroes padding, renormalizes valid weights and rejects an all-padded
recording. The window encoder now gathers valid windows **before encoding** and
scatters their embeddings back; padded windows cannot enter trainable BatchNorm.
A fully frozen encoder runs in inference mode. Partial fine-tuning keeps the
declared last encoder block trainable. No `safe_mode=False` or unsafe Lambda
deserialization is used. Variant A's pooling Lambda is also replaced.

Separable convolutions explicitly regularize both depthwise and pointwise kernels
using the supported Keras API. This documents the modernization instead of
claiming bit-identical behavior for the old unsupported `kernel_regularizer`
argument. Original window endpoints, two-stage objectives and recording-level
wrap padding are retained for the LPGNet/Variant-A training pipelines. Inference
on those models uses full unpadded recordings. ResMIL masks bag padding instead.

Bounded-memory batches use `train_on_batch`/`test_on_batch`; no unsupported Keras-3
`workers`, `use_multiprocessing` or `max_queue_size` fit arguments are passed.
Only one batch of windows or recordings is materialized at a time. MIL still
requires enough GPU memory for a batch of full recording bags: reduce the declared
batch size before a new run if required, never silently mid-run.

## Persistence

`CHECKPOINT_DIR` is explicit, absolute and versioned (`keras3_v2`) to avoid
accepting older CSV-only checkpoints. Long training requires the verified
off-runtime storage preflight in `DURABLE_RUNS.md`. Colab uses an authenticated
Drive mount; Kaggle requires an authenticated rclone remote. Configure rclone
separately using a secure credential mechanism; never put tokens into notebook
cells or Git. `/kaggle/working` and Colab `/content` are not durable run stores.
The same `RUN_ID` restores verified committed state onto fresh scratch storage;
use a new `RUN_ID` for any changed scientific plan.

Once `RUN_EXPERIMENT=True`, each completed epoch writes a full `.keras` model
including optimizer state, with full history, LR/stopping counters, best epoch
and checksums. A new generation is written before an atomic state-pointer update.
Only the committed best/latest generations are retained. Interrupted epochs
restart from the previous committed epoch; absolute epoch controls batch order.
GPU stochastic bitwise equivalence across different sessions/hardware is not
promised. Completed inner assessments and outer tasks are separately atomic JSON
transactions, so restarting does not lose a whole nested search. Stored predictions
are validated against exact filenames, identities, labels and probability ranges.

The immutable plan includes data hashes, folds, search space, training settings,
source hash and runtime versions. Changed plans are rejected; use a new directory
rather than merging old runs. Checkpoint corruption fails visibly. Reporting CSVs
are exports, **not** the resume authority. Do not edit them to change completion.
The explicit checkpoint-bundle action archives models, plan, histories and atomic
task records. Final analysis/export cells are guarded until all requested outer
tasks are complete. `MAX_NEW_TASKS` may limit work per invocation without changing
the scientific plan. Do not inspect partial outer scores for parameter selection.

## Inputs and provenance

Downloads use the existing pinned PhysioNet checksum manifest/cache instead of
unverified ZIP extraction. Clinical TXT/HTML are independently checked; HY is
joined by exact study-qualified ID and missing/conflicting values remain unknown.
The four ambiguous IDs are excluded only under the explicit curated policy.
No severity is inferred from filenames and no clinical field enters training.

## Verification and limits

The archived factorial run records Python 3.13.16, TensorFlow 2.21.0, Keras
3.13.2, NumPy 2.4.6, SciPy 1.16.3, and scikit-learn 1.6.1. The corresponding
core pins are in `requirements-factorial-archived.txt`. Tests cover model
building, backpropagation, safe serialization with optimizer state, valid-window
padding invariance, all-padded rejection, simulated interruption/recovery, task
integrity, and a small nested training task when TensorFlow is available.
The notebook smoke check must also pass in the actual Kaggle/Colab environment.
A local test or skip is not a completed GPU cohort experiment.