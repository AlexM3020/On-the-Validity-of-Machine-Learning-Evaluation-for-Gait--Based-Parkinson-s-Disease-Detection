# Long-run safety and realistic compute budgets

## What went wrong

Local atomic checkpoints prevent partially written epochs/tasks. They cannot
survive deletion of a Colab VM or Kaggle session filesystem. The former notebooks
defaulted to ephemeral working folders (or RAM-only progress for the PyTorch LPR
extensions). A kernel restart, browser disconnection and full runtime reset are
different events. If the deleted VM held the only copy, these changes cannot
recover that old run. Do not delete any still-existing old run directories.

## Required durable destination

The four maintained notebooks now require a successful upload/download verification
before starting long training. There is no notebook option to call ephemeral local
storage durable.

* **Colab:** the storage cell mounts Drive and writes versioned checkpoint objects
  under `MyDrive/gait-checkpoints/<RUN_ID>`. It verifies `/content/drive` is a real
  mount, not merely a similarly named local directory. Authenticate when Colab
  prompts you. Local compute remains on fast scratch storage.
* **Kaggle:** set `STORAGE_BACKEND='rclone'` and use an existing authenticated
  rclone remote/folder. Configure rclone separately using a secure credential
  mechanism; never put tokens into notebook cells or Git. Internet must be enabled.
  No remote access means training refuses to start. `/kaggle/working` alone is
  insufficient. This repository does not configure accounts or store secrets.

Keep the SAME RUN_ID, profile, data, code revision and compatible environment to
resume. On a new runtime the storage cell creates a fresh scratch directory,
checks cloud access and automatically restores the latest committed state for
every epoch/trial/task. One active runtime per RUN_ID; parallel writers to the
same destination are unsupported. Use a NEW RUN_ID for a changed experiment.

Each checkpoint unit is a compact uncompressed ZIP containing only its required
state/model files. A remote object is uploaded and read back to verify SHA256;
only then is a small commit marker written and read back. The latest and previous
generations are kept. A failed upload leaves the preceding valid commit recoverable
and raises an error rather than letting training proceed unprotected. A corrupt
latest snapshot fails visibly instead of silently rolling back. Runtime loss
during an uncommitted epoch can lose that epoch, not all earlier verified commits.
Read-back verification is evidence of storage access, not a guarantee against
provider outages, delayed mount synchronization, quota exhaustion or account loss.

TensorFlow snapshots contain model/optimizer, LR and stopping counters, best/latest
weights and histories. PyTorch LPR snapshots now also preserve both stages'
optimizer/scheduler, best weights and torch/loader RNG states. Completed trials
and outer tasks persist automatically; manual ZIP exports are optional archives,
not the sole survival mechanism. Unfinished epochs restart. Results are never
uploaded to GitHub by the training notebooks.

## Why tiny models can take many hours

The full factorial protocol used for the manuscript is NOT four fits:

* 4 conditions x 10 outer folds = 40 two-stage fits = **80 training stages**.
* Epoch ceilings 1,000 + 200 allow **48,000 total epochs** before early stopping.
* 50%-overlapping 2-second windows create many batches from each recording.
* Full-recording validation can be much longer than each training window.
* The frozen backbone was unnecessarily rerun in every second-stage epoch.
* Nested Variant A/ResMIL adds candidates x inner folds x training seeds,
  followed by outer-development epoch selection and a final refit.

There is no verified timing trace from the lost Colab run, so we cannot attribute
its full 10 hours to one cause or guarantee a new wall-clock duration. Logs now
separate training, validation and remote checkpoint time. GPU preflight prevents
accidental CPU research runs in the TensorFlow notebooks.

## Explicit scientific profiles (no silently dropped folds)

| Profile | Factorial notebook | Variant A / ResMIL |
|---|---|---|
| `efficient` | local/inner condition only, 10 folds | 1 seed, 3 candidates, 3 inner folds, 10 outer folds |
| `factorial_fast` | all four conditions, lower epoch ceilings | not applicable |
| `fixed` | local/inner only | one prespecified configuration, no redundant inner configuration contest; retain outer-development epoch selection and refit |
| `full` (clean factorial notebook default) | manuscript's four conditions and original epoch ceilings | previous candidate/seed and epoch budgets |

Efficient/fast/fixed ceilings are 100 encoder + 40 second-stage epochs, stopping
patience 12 and plateau patience 5. Full retains 1,000/200 and 25/14. Early stopping
still applies. Candidate spaces stay explicit. All profiles retain ten outer folds
and participant separation. Fewer candidates/seeds reduce selection robustness;
they are declared budget changes, not evidence of equivalence to a larger search.

`efficient` in the factorial notebook is a corrected-condition experiment,
**not a completed 2x2 factorial and not an exact LPGNet replication**. No missing
condition contrasts are invented. Select `full` for the original compute protocol,
or `factorial_fast` for all four conditions at the smaller declared budget.
The manuscript's factorial reconstruction uses `official_all`, which retains
known duplicate identities. Selecting `curated` defines a different experiment
and requires a new run ID; it must not be substituted into the reported plan.

The PyTorch LPR notebook defaults to both extensions, ten folds, one seed and two
candidates: **40 two-stage fits**, rather than 180. Its 60/40 epoch caps are unchanged.
Both arms and the same outer cohort remain present. The larger three-seed,
three-candidate plan remains an explicit option under a new run ID.

## Faster frozen stages and LP reuse

The accelerated TensorFlow profile caches deterministic frozen-encoder features
once for the training and validation partitions. Only the small head is evaluated
on every subsequent stage-2 epoch. MIL caching applies ONLY when its encoder is
fully frozen; partial fine-tuning still runs the encoder normally. Fixed frozen
L2 penalties are preserved in the cached head, and inference probability/loss
equivalence is regression-tested. LP representations are reused across inner
candidates with the same fitting partition using a bounded in-memory cache.

For LPGNet/Variant A the original frozen encoder can still contain stochastic
noise/dropout active in training mode. The accelerated profile fixes it in
inference mode for caching. This is an explicit training-protocol change;
`full` disables this caching and preserves the uncached path. Do not label the
accelerated path exact stochastic replication. Cached feature loss/prediction
parity is measured against the deterministic full encoder, not a stochastic pass.

A synthetic CPU check with 8 full 6,000-step recordings measured 20 stage-2 batch
steps at about 0.58 s uncached versus 0.032 s cached, after ~2.7 s one-time cache
construction. This demonstrates the avoided computation, not an end-to-end GPU
speed claim. Real gains depend on epochs, devices, shape tracing and cloud I/O.

## Tests and safe migration

Tests simulate failed object/commit uploads, corruption, complete deletion of
local runtime state, TensorFlow model/optimizer recovery and PyTorch stage-2/RNG
recovery. Separate cloud provider authentication/mount behavior must be verified
in your notebook by the mandatory preflight; no live credentials are available
in the local test environment.

New source hashes/profiles require new run IDs. Old checkpoints remain untouched
and can only resume with their old pinned notebook/configuration if still present.
An old optional archive cannot be merged into this run without an explicit
validated migration. Reduced-cost results must not be pooled with old full-profile
results as though the training procedure were identical.