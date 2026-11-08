# Subject-Independent Evaluation of Mamba for Parkinson's Disease Detection from Gait Signals

Anonymous research artifact for binary gait classification with Mamba, LSTM and
GRU. This release preserves **existing results and historical source**, and adds
an independently written filename/split audit and a static checkpoint inspector.
It is **not an exact training reproduction**, a new benchmark, or clinical software.
No training or new predictive results were produced for this artifact.

## Anonymous TMLR manuscript and Overleaf

Upload **[paper/main.tex](paper/main.tex)** and the entire
**[paper/figures/](paper/figures/)** directory to an empty Overleaf project. Select
pdfLaTeX and set the main document to `main.tex`. The folder contains only this
TeX file and three figure assets: a newly drawn vector architecture and the two
original loss plots. Bibliography entries and the unmodified official TMLR style
are embedded in the TeX source; no separate `.bib` or `.sty` upload is required.
The style is generated automatically at compilation. Keep the default anonymous
submission mode, not the `accepted` or `preprint` option.

**Before submission**, create the anonymous repository mirror and insert its
actual tested URL into `\AnonymousCodeURL` in the TeX preamble. The current draft
prints a visible placeholder. No anonymous mirror has been created and no URL
has been invented. Do not put an identifying GitHub address in that field.
Although content and commit authorship are anonymous, an ordinary GitHub account
URL is not itself anonymous. Check mirror settings, visible files, external links,
and expiry/redirect behavior before sharing it with reviewers.

The manuscript is a bounded empirical/audit draft, not a claim of being the only
subject-independent method. It keeps missing cross-validation, unmatched sample
supports, and other provenance limits explicit. Confirm original-work eligibility
and required human authorship/ethics information separately in the submission
system. Nothing has been submitted to a journal.

Optional figure regeneration uses [tools/make_figures.py](tools/make_figures.py)
and [requirements-figures.txt](requirements-figures.txt). Regenerating the
architecture requires no training data. Original loss-image arguments are
optional; when provided, decoded pixels are verified unchanged and image metadata
is removed. The architecture is a documented-design schematic, not an exact
reconstruction of the historical forward pass.

The embedded official template comes from `JmlrOrg/tmlr-style-file` revision
`7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`. Its Apache-2.0 license is provided in
[LICENSE-TMLR.txt](LICENSE-TMLR.txt); this license applies to the third-party
template, not automatically to the original experiment code or manuscript.

## Existing results

The three result files are unmodified copies of the original saved reports.
They retain their original line endings and are exempt from the otherwise-LF
text normalization in [.gitattributes](.gitattributes), preserving exact hashes.
Support counts refer to scored samples, **not independent participants**.

| Saved report | Accuracy | Macro-F1 | Support |
| --- | ---: | ---: | ---: |
| [results/mamba_report.json](results/mamba_report.json) | 88.61% | 88.52% | 790 |
| [results/lstm_report.csv](results/lstm_report.csv) | 87.18% | 87.16% | 390 |
| [results/gru_report.csv](results/gru_report.csv) | 84.36% | 84.34% | 390 |

These are validation results from a model-selection workflow, not an untouched
test-set comparison. The unequal supports prevent an assumption of matched
validation inputs. A small numerical advantage does not establish significance,
architecture superiority, clinical generalization, or more stable optimization.
The CSV accuracy row retains the original tabular-export convention; use class
supports or the macro-average support, not its accuracy-row support cell.

## Audit without training or dependencies

Use Python 3.10+ and run these commands from this directory. The utilities and
tests use only the standard library; the tests were checked with Python 3.14.6.

```console
python -B audit_splits.py
python -B audit_splits.py --fold 1
python -B -m unittest discover -s tests -v
```

The default audit is offline and reads [fixtures/SHA256SUMS.txt](fixtures/SHA256SUMS.txt).
The original manifest was fetched publicly from PhysioNet, without sensor data;
its URL, exact SHA-256 and retrieval date are recorded in
[fixtures/manifest_provenance.json](fixtures/manifest_provenance.json).
The captured manifest contains **306 matching sensor files, 165 study-qualified
participant IDs, and 54 IDs with multiple recordings**. These filename counts
do not resolve the dataset description's advertised 166-person cohort.

Network access is explicitly opt-in and fetches only the small manifest into
memory, without replacing the bundled fixture:

```console
python -B audit_splits.py --download-manifest
```

A custom local manifest can be supplied with `--manifest`. Successful output is
JSON on stdout, including manifest checksum, recording filenames, participant
IDs, partitions and overlap evidence. Operational errors return status 2 and a
JSON error on stderr. A downloaded manifest with a different digest is explicitly
marked as not matching the bundled snapshot; it is not silently treated as identical.

### Scope of the control-branch replay

The released implementations associated with El Maachi et al. and Naimi et al.
shuffle sorted **recording files** and maintain a cumulative segment boundary for
each recording. The audit independently expresses their **control-branch only**
fold arithmetic. For `N` control files, there are `N+1` boundaries, including zero:
`width = int((N + 1) * (1 / 10))`; fold `k` selects recording indices
`[k * width : (k + 1) * width]`. This assumes each listed file contributes positive
segments. No waveform lengths, patient-branch offsets, severity labels, neural
networks, or upstream source code are loaded or copied.

The audited source revisions are `c5a6d5f1e2a8a53cc92f020b675fa6ad7243c372`
(El Maachi et al.) and `fb1ace4e6c038a87674bf8ab0610055a19737d2f` (Naimi et al.).
References: [El Maachi et al.](https://doi.org/10.1016/j.eswa.2019.113075) and
[Naimi et al.](https://doi.org/10.1007/s00521-023-09193-6).

**Seed 42 is an arbitrary audit seed, not the original authors' seed.** A fresh
Python RNG is used for each subset. With the bundled manifest and fold zero:

| Control subset | Input files | Validation files | Overlapping participant IDs |
| --- | ---: | ---: | ---: |
| El Maachi: all controls | 92 | 9 | 4 |
| Naimi: Ga controls only | 38 | 3 | 3 |

Concrete examples are validation `GaCo02_01` versus training `GaCo02_02`, and
validation `GaCo06_02` versus training `GaCo06_01`, respectively. Participant
identity retains the full study prefix, class and two-digit number; recording
and chunk suffixes do not create new participants.

An independent grouped counterexample shuffles unique participant IDs, assigns
them round-robin to ten folds, and keeps all their recordings together. It has
**zero participant overlap**, including all ten tested folds. It is not presented
as a recreation of the original experiments or the only valid evaluation design.
The historical integer-boundary formula is not silently repaired or replaced by
ordinary k-fold splitting; small inputs can produce empty/out-of-range folds.

These examples demonstrate that file grouping does not enforce participant
disjointness. They do **not** establish which folds generated published scores,
how much any score was inflated, or that high accuracy is pure memorization.
A matched split-only training ablation would be future work, not a result here.

## Historical scripts, deliberately not repaired

| Snapshot | Status |
| --- | --- |
| [legacy/lstm.py](legacy/lstm.py) | Historical binary subject-grouping code; not the exact generating run |
| [legacy/gru.py](legacy/gru.py) | Historical binary subject-grouping code; not the exact generating run |
| [legacy/mamba.py](legacy/mamba.py) | Historical incompatible/different model definition; not the exact generating run |
| [legacy/lstm2.py](legacy/lstm2.py) | File-level split with participant-leakage risk; **not a paper-validated result** |
| [legacy/lstm_multi_class.py](legacy/lstm_multi_class.py) | Historical multiclass variant; **no paper-validated multiclass result supplied** |

Only identifying input paths/comments were replaced and warning docstrings added.
Model computation, splitting, preprocessing, output filenames and historical bugs
remain unchanged. Inputs are selected through `GAIT_DATA_DIR` (default: local raw
data directory), `GAIT_FEATURES_CSV`, and `GAIT_LABELS_CSV`. The feature CSV is still
read by the recurrent binary scripts even when selected features are empty.
These snapshots originally expected prepared/chunked inputs, sometimes oversampled;
pointing them at official raw recordings is not an equivalent historical run.
Their subject parser takes the first six filename characters and relies on naming
conventions; the new audit parser is not substituted into historical code.

**Do not import the historical Mamba script:** it executes training and writes
outputs at import time. It passes unsupported `n_layers` to a single Mamba block,
and selects the final timestep rather than implementing the report's temporal
max pooling. Both discrepancies remain visible. Legacy recurrent scripts also
retain full-model pickle loading; never load untrusted checkpoints with them.

[requirements-legacy.txt](requirements-legacy.txt) lists unpinned reference
dependencies, **not a tested historical lockfile**. No legacy execution or GPU
compatibility is certified. No new training runner is supplied.

## Configuration and provenance limits

[configurations/report_configuration.json](configurations/report_configuration.json)
separates report settings, run notes, local static checkpoint observations, result
hashes and source discrepancies. In particular:

- The report describes 2,000 timesteps; the run-10 note says 1,000.
- The run note says `d_conv=16`; checkpoint metadata and the report table say 32.
- The report's two-block model is not the preserved script's single-block call.
- LSTM batch size/channels and GRU width/learning rate differ between report and
  source. Loss, learning rate and batch size also confound architecture comparisons.
- The report calls the 80% portion "testing"; historical code trains on that
  portion. An independently held-out test set is not established.
- Exact split membership, augmentation lineage, the generating source revision,
  and complete training random states are not available. Subject-grouping code
  alone cannot certify the provenance of the saved reports.

No original course report, slides, raw signals, feature tables, full-model pickles, checkpoints,
vendored Mamba library, or identifying project links are included. No license for
the original project code is asserted or invented here; third-party data/software
remain subject to their own terms.

## Nonexecuting checkpoint inspection

[inspect_checkpoint.py](inspect_checkpoint.py) accepts a separately supplied ZIP
checkpoint, reads only its small pickle-metadata member, and reports archive
metadata plus literal global/string/integer opcode tokens as JSON. It never
unpickles, imports saved globals, extracts an archive, or reads tensor payloads.
For example, set `CHECKPOINT` to your own local file and run:

```console
python -B inspect_checkpoint.py "$CHECKPOINT"
```

`STACK_GLOBAL` occurrences are counted but deliberately unresolved. Tokens are
not reconstructed attributes; forward, residual and pooling behavior cannot be
recovered from serialized module metadata alone. Size/opcode limits are present,
but this is not malware certification. Legacy non-ZIP checkpoints are rejected.
**Inspector output can contain saved names or paths; review it before sharing.**

Dataset source: [PhysioNet Gait in Parkinson's Disease](https://physionet.org/content/gaitpdb/1.0.0/).
