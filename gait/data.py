"""Subject-independent data preparation for a reconstructed gait evaluation.

Split recordings by participant before constructing windows. Normalization is
per recording, not fitted on other participants. Paths retain the caller's
root convention; a runner can serialize them relative to its data root.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import re
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_RECORDING_NAME = re.compile(
    r"(?P<subject>(?:Ga|Ju|Si)(?P<group>Co|Pt)[0-9]{2})"
    r"(?:_[0-9]+)?(?:_chunk_[0-9]+)?\.txt"
)
DEFAULT_SENSOR_COLUMNS = tuple(range(1, 19))


@dataclass(frozen=True)
class Recording:
    """A sensor file with its full study/class/participant identifier."""

    path: Path
    subject: str
    label: int


@dataclass(frozen=True)
class Window:
    """Window provenance; valid_length excludes synthetic trailing padding."""

    record: Recording
    start: int
    valid_length: int


def discover_recordings(root: str | Path) -> list[Recording]:
    """Discover strict sensor names recursively (Co=0, Pt=1).

    Accepted examples include GaCo01.txt, GaCo01_01.txt, and
    GaCo01_01_chunk_1.txt. Non-sensor text files are ignored. Repeated
    basenames, even in different directories, are rejected as ambiguous.
    Discovery inspects names only and does not read sensor values.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Recording directory does not exist: {root}")
    recordings: list[Recording] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*.txt")):
        match = _RECORDING_NAME.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        if path.name in seen:
            raise ValueError(f"Duplicate recording filename: {path.name}")
        seen.add(path.name)
        recordings.append(
            Recording(path, match.group("subject"), int(match.group("group") == "Pt"))
        )
    if not recordings:
        raise ValueError("No sensor recordings with supported filenames were found")
    return recordings


def _fraction(value: float, name: str, *, allow_zero: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not (0 <= value < 1 if allow_zero else 0 < value < 1)
    ):
        raise ValueError(f"{name} must be finite and {'in [0, 1)' if allow_zero else 'in (0, 1)'}")


def _check_recording(record: Recording) -> None:
    if not isinstance(record.path, Path) or not record.subject:
        raise ValueError("Recording requires a Path and a nonempty subject identifier")
    if isinstance(record.label, bool) or not isinstance(record.label, Integral) or record.label not in (0, 1):
        raise ValueError("Recording labels must be integer 0 (Co) or 1 (Pt)")


def split_recordings(
    recordings: Iterable[Recording],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, list[Recording]]:
    """Stratify unique subjects, then assign all their recordings/chunks.

    Fractions refer to the full set of subjects. With a test split, test
    subjects are selected first; validation uses validation_fraction /
    (1 - test_fraction) of the remainder. Small strata may be unsplittable;
    in that case raise rather than silently falling back to file-level splits.
    The optional ``test`` key is present only when test_fraction is positive.
    """
    _fraction(validation_fraction, "validation_fraction", allow_zero=False)
    _fraction(test_fraction, "test_fraction", allow_zero=True)
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave training subjects")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    records = list(recordings)
    if not records:
        raise ValueError("Cannot split an empty collection of recordings")
    subject_labels: dict[str, int] = {}
    for record in records:
        _check_recording(record)
        if record.subject in subject_labels and subject_labels[record.subject] != record.label:
            raise ValueError(f"Conflicting labels for subject {record.subject}")
        subject_labels[record.subject] = record.label
    subjects = sorted(subject_labels)

    # Dependency is needed only for splitting, never for filename discovery.
    from sklearn.model_selection import train_test_split

    def stratified(ids: list[str], fraction: float) -> tuple[list[str], list[str]]:
        return train_test_split(
            ids,
            test_size=fraction,
            random_state=int(seed),
            stratify=[subject_labels[subject] for subject in ids],
        )

    try:
        remaining, test = stratified(subjects, test_fraction) if test_fraction else (subjects, [])
        train, validation = stratified(remaining, validation_fraction / (1 - test_fraction))
    except ValueError as exc:
        raise ValueError(f"Cannot stratify subjects with the requested fractions: {exc}") from exc
    groups = {"train": set(train), "validation": set(validation)}
    if test_fraction:
        groups["test"] = set(test)
    memberships = list(groups.values())
    for index, left in enumerate(memberships):
        for right in memberships[index + 1:]:
            assert left.isdisjoint(right), "Subject leakage across partitions"
    assert set().union(*memberships) == set(subjects), "Subject split lost recordings"
    records.sort(key=lambda record: (record.subject, str(record.path)))
    return {
        partition: [record for record in records if record.subject in ids]
        for partition, ids in groups.items()
    }


class GaitDataset(Dataset):
    """Eagerly cached, per-recording z-scored, nonoverlapping gait windows.

    Default zero-based columns 1..18 exclude time column 0. Normalize all
    selected rows before windowing, using population standard deviation.
    Constant channels become zero. For long recordings discard the partial
    final window; for a short recording emit one zero-padded window.
    ``windows`` and ``recordings`` retain provenance, not partition decisions.
    """

    def __init__(
        self,
        recordings: Iterable[Recording],
        seq_len: int = 2000,
        sensor_columns: Sequence[int] = DEFAULT_SENSOR_COLUMNS,
    ) -> None:
        if isinstance(seq_len, bool) or not isinstance(seq_len, Integral) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        columns = tuple(sensor_columns)
        if not columns or any(
            isinstance(column, bool) or not isinstance(column, Integral) or column < 0
            for column in columns
        ) or len(set(columns)) != len(columns):
            raise ValueError("sensor_columns must be distinct nonnegative integer indices")
        self.seq_len = int(seq_len)
        self.sensor_columns = columns
        self.recordings = list(recordings)
        self.windows: list[Window] = []
        self._signals: dict[Path, np.ndarray] = {}
        for record in self.recordings:
            _check_recording(record)
            if record.path not in self._signals:
                try:
                    raw = np.loadtxt(record.path, dtype=np.float64, ndmin=2)
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Cannot read sensor matrix {record.path}: {exc}") from exc
                if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] <= max(columns):
                    raise ValueError(f"Invalid sensor matrix shape {raw.shape} in {record.path}")
                if not np.isfinite(raw).all():
                    raise ValueError(f"Nonfinite values in sensor matrix {record.path}")
                signals = raw[:, columns]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    mean = signals.mean(axis=0)
                    std = signals.std(axis=0)
                    normalized = (signals - mean) / np.where(std == 0, 1.0, std)
                if not (
                    np.isfinite(mean).all()
                    and np.isfinite(std).all()
                    and np.isfinite(normalized).all()
                ):
                    raise ValueError(f"Nonfinite channel normalization in {record.path}")
                self._signals[record.path] = normalized.astype(np.float32)
            count = len(self._signals[record.path])
            if count < self.seq_len:
                self.windows.append(Window(record, 0, count))
            else:
                self.windows.extend(
                    Window(record, start, self.seq_len)
                    for start in range(0, count - self.seq_len + 1, self.seq_len)
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        values = np.zeros((self.seq_len, len(self.sensor_columns)), dtype=np.float32)
        values[:window.valid_length] = self._signals[window.record.path][
            window.start:window.start + window.valid_length
        ]
        return (
            torch.from_numpy(values),
            torch.tensor(window.record.label, dtype=torch.long),
            torch.tensor(window.valid_length, dtype=torch.long),
        )
