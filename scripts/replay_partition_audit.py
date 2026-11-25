"""Deterministic metadata-only replay of the released recording-fold paths.

This script downloads only the checksum-pinned PhysioNet SHA256SUMS.txt manifest.
It does not fetch waveforms, execute third-party training code, or reconstruct the
original publications' random states. Audit seed 42 and fold zero are deliberate
replay choices described in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold

# Allow execution as ``python scripts/replay_partition_audit.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait.official_data import MANIFEST_SHA256, RECORDING, manifest  # noqa: E402

AUDIT_SEED = 42


def participant(filename: str) -> str:
    """Return the exact study-qualified participant ID, including Co/Pt."""
    return filename[:6]


def overlap(files: list[str], held_out: list[int] | np.ndarray) -> int:
    held = {participant(files[int(i)]) for i in held_out}
    held_indices = {int(i) for i in held_out}
    train = {participant(name) for i, name in enumerate(files) if i not in held_indices}
    return len(held & train)


def grouped_overlap(files: list[str], n_splits: int) -> int:
    """Allocate complete participant groups; the same participants define all folds."""
    participants = sorted({participant(name) for name in files})
    rng = np.random.default_rng(AUDIT_SEED)
    order = rng.permutation(participants)
    groups = np.array_split(order, n_splits)
    held = set(groups[0])
    train = set(participants) - held
    assert not held & train
    # Apply the assignment to the recording inventory as an explicit invariant check.
    held_files = {name for name in files if participant(name) in held}
    train_files = {name for name in files if participant(name) in train}
    assert not {participant(n) for n in held_files} & {participant(n) for n in train_files}
    return 0


def control_replay(files: list[str]) -> tuple[int, int]:
    shuffled = list(files)
    random.Random(AUDIT_SEED).shuffle(shuffled)
    # One cumulative segment boundary is appended per recording. With at least one
    # segment per recording, N recordings produce N+1 boundaries.
    q = (len(shuffled) + 1) // 10
    held_out = list(range(0, q))
    return len(held_out), overlap(shuffled, held_out)


def subgaitnet_replay(files: list[str]) -> tuple[int, int]:
    # Released path: initial NumPy shuffle, followed by shuffled five-fold KFold.
    shuffled = list(files)
    np.random.seed(AUDIT_SEED)
    np.random.shuffle(shuffled)
    train, test = next(KFold(n_splits=5, shuffle=True, random_state=AUDIT_SEED).split(shuffled))
    assert set(train).isdisjoint(test)
    return len(test), overlap(shuffled, test)


def replay_audit(cache: str | Path) -> list[dict[str, object]]:
    entries = manifest(Path(cache), download=True)
    names = sorted(name for name in entries if RECORDING.fullmatch(name))
    if len(names) != 306:
        raise ValueError("Unexpected official recording inventory")
    controls = [name for name in names if name[2:4] == "Co"]
    ga_controls = [name for name in controls if name.startswith("Ga")]
    if (len(controls), len(ga_controls)) != (92, 38):
        raise ValueError("Unexpected official control subsets")

    rows = []
    specifications = [
        ("elmaachi2020", "all controls", controls),
        ("naimi2024", "Ga controls", ga_controls),
        ("nguyen2022", "all controls", controls),
        ("naimi2023hct", "all controls", controls),
        ("naimi2025inceptoformer", "Ga controls", ga_controls),
        ("zhang2025benchmark", "all controls", controls),
    ]
    for study, subset, files in specifications:
        held_out, id_overlap = control_replay(files)
        rows.append({
            "study": study,
            "subset": subset,
            "files": len(files),
            "held_out": held_out,
            "id_overlap": id_overlap,
            "grouped_overlap": grouped_overlap(files, 10),
        })
    held_out, id_overlap = subgaitnet_replay(names)
    rows.append({
        "study": "li2026subgaitnet",
        "subset": "all recordings",
        "files": len(names),
        "held_out": held_out,
        "id_overlap": id_overlap,
        "grouped_overlap": grouped_overlap(names, 5),
    })

    expected = [
        (92, 9, 4), (38, 3, 3), (92, 9, 4), (92, 9, 4),
        (38, 3, 3), (92, 9, 4), (306, 62, 29),
    ]
    actual = [(int(r["files"]), int(r["held_out"]), int(r["id_overlap"])) for r in rows]
    if actual != expected:
        raise AssertionError(f"Audit replay differs from the manuscript: {actual!r}")
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/physionet"),
                        help="Checksum-verified PhysioNet metadata cache")
    parser.add_argument("--output", type=Path, default=Path("generated_paper/tab-audit-replay.csv"))
    args = parser.parse_args()
    rows = replay_audit(args.cache)
    write_csv(rows, args.output)
    print(f"Manifest SHA-256: {MANIFEST_SHA256}")
    print(f"Wrote {args.output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
