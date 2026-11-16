"""Nested participant holdout selection inside stratified outer test folds.

No data downloads, training, exports, or dependency installation on import.
Research results require the real CUDA Mamba backend. Tests explicitly mock it.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import time
import zipfile

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import build_model


@dataclass(frozen=True)
class Protocol:
    seq_len: int = 2000
    overlap: float = 0.0
    outer_folds: int = 10
    inner_fraction: float = 0.2
    split_seed: int = 42
    training_seeds: tuple = (42,)
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-7
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 7301
    num_workers: int = 0


# A small prespecified search; every candidate keeps the same architecture.
DEFAULT_CANDIDATES = [
    {"name": "reference", "lr": 5e-5, "weight_decay": 0.0},
    {"name": "lower_lr", "lr": 2.5e-5, "weight_decay": 0.0},
    {"name": "mild_l2", "lr": 5e-5, "weight_decay": 1e-4},
]
ARCHITECTURE = dict(input_dim=18, hidden_dim=128, num_layers=2,
                    d_state=32, d_conv=32, expand=2, dropout=0.0)


def validate_protocol(cfg, candidates):
    for field in ("seq_len", "outer_folds", "batch_size", "max_epochs", "patience",
                  "scheduler_patience", "bootstrap_samples"):
        value = getattr(cfg, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if cfg.outer_folds < 2 or not 0 <= cfg.overlap < 1 or int(cfg.seq_len * (1-cfg.overlap)) < 1:
        raise ValueError("Need at least two folds and a positive window stride")
    if not 0 < cfg.inner_fraction < 1 or not 0 < cfg.scheduler_factor < 1:
        raise ValueError("Invalid inner fraction or scheduler factor")
    if not np.isfinite(cfg.min_lr) or cfg.min_lr <= 0 or cfg.num_workers < 0:
        raise ValueError("Invalid min_lr or worker count")
    seeds = (cfg.split_seed, cfg.bootstrap_seed, *cfg.training_seeds)
    if not cfg.training_seeds or len(set(cfg.training_seeds)) != len(cfg.training_seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    if not candidates or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidates must have unique names")
    for c in candidates:
        if set(c) != {"name", "lr", "weight_decay"} or not c["name"]:
            raise ValueError("Candidates specify name, lr and weight_decay only")
        if not np.isfinite(c["lr"]) or c["lr"] < cfg.min_lr:
            raise ValueError("Candidate learning rate must be >= min_lr")
        if not np.isfinite(c["weight_decay"]) or c["weight_decay"] < 0:
            raise ValueError("Invalid weight decay")


def load_originals(data_dir, seq_len=2000):
    """Audit original 19-column files; deduplicate identical signals within identity.

    Byte and numeric-force hashes are retained. Identical signals attributed to
    different identities/labels fail closed. Copy-name variants are recognized;
    chunk-suffixed/prepared inputs are rejected rather than silently included.
    Tail/short exclusions are explicit, and no samples are padded or augmented.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")
    pattern = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d{2})_(\d{2})(?: \(\d+\)| - Copy)?\.txt$", re.I)
    records, audit, by_force, by_name = [], [], {}, {}
    for path in sorted(root.rglob("*.txt")):
        match = pattern.fullmatch(path.name)
        if match is None:
            if re.match(r"^(Ga|Ju|Si)(Co|Pt)\d", path.name, re.I):
                raise ValueError(f"Prepared/unsupported recording name: {path.name}; use original recordings")
            continue
        study, group, sid, walk = match.groups()
        study, group = study.title(), group.title()
        subject = study + group + sid
        canonical = subject + "_" + walk
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)
        if values.shape[1] != 19 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Expected finite original 19-column recording: {path}")
        dt = np.diff(values[:, 0])
        if not np.all(dt > 0) or not np.allclose(dt, 0.01, atol=0.002, rtol=0):
            raise ValueError(f"Non-monotonic or non-100-Hz time axis: {path}")
        forces = np.ascontiguousarray(values[:, 1:], dtype="<f8")
        force_hash = hashlib.sha256(forces.tobytes()).hexdigest()
        row = dict(path=path.relative_to(root).as_posix(), recording=canonical,
                   subject=subject, label=int(group == "Pt"), study=study,
                   samples=len(values), sha256=raw_hash, force_sha256=force_hash)
        if canonical in by_name and by_name[canonical] != force_hash:
            raise ValueError(f"Conflicting contents for original recording {canonical}")
        by_name[canonical] = force_hash
        if force_hash in by_force:
            previous = by_force[force_hash]
            if previous["subject"] != subject:
                raise ValueError(f"Identical signals assigned to different participants: {previous['path']} / {row['path']}")
            row.update(status="duplicate_excluded", duplicate_of=previous["path"])
        else:
            by_force[force_hash] = row
            row["status"] = "included" if len(forces) >= seq_len else "short_excluded"
            if row["status"] == "included":
                with np.errstate(over="ignore"):
                    signal = forces.astype(np.float32)
                if not np.isfinite(signal).all():
                    raise ValueError(f"Forces exceed float32 range: {path}")
                records.append({**row, "signal": signal})
        audit.append(row)
    if not records:
        raise ValueError("No eligible original recordings")
    return records, audit


def make_windows(records, seq_len, overlap=0.0):
    stride = int(seq_len * (1-overlap))
    if seq_len < 1 or not 0 <= overlap < 1 or stride < 1:
        raise ValueError("Invalid sequence length/overlap")
    windows = []
    for index, record in enumerate(records):
        for start in range(0, len(record["signal"])-seq_len+1, stride):
            windows.append(dict(record_index=index, recording=record["recording"],
                                subject=record["subject"], label=record["label"],
                                start=start, stop=start+seq_len))
    if not windows:
        raise ValueError("No complete windows")
    return windows


def make_splits(records, cfg):
    labels = {}
    for r in records:
        if r["subject"] in labels and labels[r["subject"]] != r["label"]:
            raise ValueError("Inconsistent participant labels")
        labels[r["subject"]] = r["label"]
    subjects = np.array(sorted(labels))
    y = np.array([labels[s] for s in subjects])
    if set(y) != {0, 1} or np.bincount(y, minlength=2).min() < cfg.outer_folds:
        raise ValueError("Each class needs at least outer_folds unique participants")
    outer = StratifiedKFold(cfg.outer_folds, shuffle=True, random_state=cfg.split_seed)
    splits = []
    for fold, (development, test) in enumerate(outer.split(subjects, y), 1):
        training, validation = train_test_split(
            development, test_size=cfg.inner_fraction,
            stratify=y[development], random_state=(cfg.split_seed+fold) % 2**32,
        )
        groups = dict(train=subjects[training].tolist(), validation=subjects[validation].tolist(),
                      test=subjects[test].tolist())
        sets = [set(v) for v in groups.values()]
        if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
            raise AssertionError("Participant overlap")
        for group in groups.values():
            if {labels[s] for s in group} != {0, 1}:
                raise ValueError("Every partition must contain both classes")
        splits.append(dict(fold=fold, **groups))
    tested = [s for split in splits for s in split["test"]]
    if len(tested) != len(set(tested)) or set(tested) != set(subjects):
        raise AssertionError("Outer test coverage must be exactly once per participant")
    return splits


def fit_normalizer(records, train_subjects):
    """Streaming float64 moments over each original inner-training sample once."""
    total = 0
    mean, m2 = np.zeros(18), np.zeros(18)
    for r in records:
        if r["subject"] not in train_subjects:
            continue
        x = r["signal"].astype(np.float64)
        n = len(x)
        delta = x.mean(0) - mean
        combined = total + n
        m2 += x.var(0)*n + delta**2 * total*n/combined
        mean += delta*n/combined
        total = combined
    if not total:
        raise ValueError("Empty normalization training set")
    std = np.sqrt(m2/total)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, records, windows, subjects, mean, std):
        self.records = records
        self.windows = [w for w in windows if w["subject"] in subjects]
        self.mean, self.std = mean, std
        if not self.windows:
            raise ValueError("Partition has no windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        w = self.windows[index]
        x = self.records[w["record_index"]]["signal"][w["start"]:w["stop"]]
        return torch.from_numpy((x-self.mean)/self.std), w["label"]


def class_weights(dataset):
    counts = np.bincount([w["label"] for w in dataset.windows], minlength=2)
    if (counts == 0).any():
        raise ValueError("Training windows must contain both classes")
    return torch.tensor(counts.sum()/(2*counts), dtype=torch.float32)


def aggregate_predictions(rows):
    """Window probability mean per recording, then equal recording mean per person."""
    by_record = {}
    for row in rows:
        key = row["subject"], row["recording"]
        by_record.setdefault(key, []).append(row)
    by_subject = {}
    for (subject, recording), group in sorted(by_record.items()):
        if len({r["label"] for r in group}) != 1:
            raise ValueError("Conflicting labels within recording")
        by_subject.setdefault(subject, []).append(dict(
            recording=recording, label=group[0]["label"],
            probability=float(np.mean([r["probability"] for r in group])), windows=len(group)))
    participants = []
    for subject, group in sorted(by_subject.items()):
        if len({r["label"] for r in group}) != 1:
            raise ValueError("Conflicting participant labels")
        p = float(np.mean([r["probability"] for r in group]))
        participants.append(dict(subject=subject, label=group[0]["label"], probability=p,
                                 prediction=int(p >= 0.5), recordings=len(group),
                                 windows=sum(r["windows"] for r in group)))
    return participants


def metrics(rows):
    y = np.array([r["label"] for r in rows])
    p = np.array([r["probability"] for r in rows])
    if len(y) == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid predictions")
    if set(y) != {0, 1}:
        raise ValueError("Binary metrics require both classes")
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                roc_auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=len(y))


def confidence_intervals(rows, n_bootstrap=2000, seed=7301):
    """Class-stratified participant bootstrap, conditional on fitted OOF predictions."""
    if len({r["subject"] for r in rows}) != len(rows):
        raise ValueError("Bootstrap requires one prediction per participant, not windows/repeated seeds")
    y = np.array([r["label"] for r in rows])
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    if any(len(g) == 0 for g in groups) or n_bootstrap < 1:
        raise ValueError("Bootstrap requires both classes and positive replicate count")
    names = ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc")
    samples = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        scores = metrics([rows[i] for i in idx])
        for name in names:
            samples[name].append(scores[name])
    point = metrics(rows)
    return {name: dict(estimate=point[name], lower=float(np.percentile(samples[name], 2.5)),
                       upper=float(np.percentile(samples[name], 97.5))) for name in names}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Real Mamba execution requires a Kaggle/Linux CUDA GPU; no CPU substitute")
    return torch.device("cuda")


@torch.no_grad()
def predict(model, dataset, batch_size, device, num_workers=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities = []
    for x, _ in loader:
        logits = model(x.to(device))
        probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    rows = [{**w, "probability": float(p)} for w, p in zip(dataset.windows, probabilities)]
    if len(rows) != len(dataset):
        raise AssertionError("Prediction alignment failed")
    return rows


def fit_candidate(train_ds, validation_ds, candidate, cfg, seed, device, verbose=True):
    """Only inner data accepted: scheduler, checkpoint and stopping use participant NLL."""
    seed_everything(seed)
    model = build_model(**ARCHITECTURE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["lr"],
                                 weight_decay=candidate["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience, min_lr=cfg.min_lr)
    weights = class_weights(train_ds).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, generator=generator, drop_last=False)
    history, best_state, best_metrics = [], None, None
    best_loss, stale, best_epoch = float("inf"), 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        numerator, denominator = 0.0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            summed = nn.functional.cross_entropy(model(x), y, weight=weights, reduction="sum")
            mass = weights[y].sum()
            loss = summed/mass
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            numerator += summed.item()
            denominator += mass.item()
        validation_rows = aggregate_predictions(predict(model, validation_ds, cfg.batch_size,
                                                        device, cfg.num_workers))
        score = metrics(validation_rows)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(score["log_loss"])
        history.append(dict(epoch=epoch, train_loss=numerator/denominator, lr=lr,
                            validation_log_loss=score["log_loss"],
                            validation_balanced_accuracy=score["balanced_accuracy"]))
        if score["log_loss"] < best_loss - 1e-6:
            best_loss, stale, best_epoch = score["log_loss"], 0, epoch
            best_metrics = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose:
            print(f"  {candidate['name']} epoch {epoch}: train CE={numerator/denominator:.4f}, "
                  f"inner participant NLL={score['log_loss']:.4f}, stale={stale}/{cfg.patience}", flush=True)
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint was selected")
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, dict(candidate=copy.deepcopy(candidate), best_epoch=best_epoch,
                            inner_metrics=best_metrics, history=history,
                            seconds=time.perf_counter()-start, class_weights=weights.cpu().tolist())


def runtime_info():
    versions = {}
    for package in ("numpy", "torch", "scikit-learn", "mamba-ssm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def run_nested_cv(records, audit, cfg=None, candidates=None, *, progress=None, verbose=True):
    """Complete all seeds/folds; no partial-run metrics are presented as complete CV.

    progress is an optional RAM dictionary updated after each completed fold so
    interrupted-session outputs remain inspectable. Nothing is exported here.
    No outer-test score is printed until ALL model selection has completed.
    """
    cfg = Protocol() if cfg is None else cfg
    candidates = copy.deepcopy(DEFAULT_CANDIDATES if candidates is None else candidates)
    validate_protocol(cfg, candidates)
    windows = make_windows(records, cfg.seq_len, cfg.overlap)
    eligible = {w["subject"] for w in windows}
    if eligible != {r["subject"] for r in records}:
        raise ValueError("Reload originals with the configured sequence length before splitting")
    splits = make_splits(records, cfg)
    # Duplicate checks cannot be bypassed by passing hand-built records.
    hashes = [r["force_sha256"] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate recordings remain in analyzed inputs")
    device = require_cuda()
    result = dict(protocol=asdict(cfg), candidates=candidates, architecture=ARCHITECTURE.copy(),
                  environment=runtime_info(), audit=copy.deepcopy(audit), splits=splits,
                  window_manifest=windows, runs=[], complete=False)
    result["source_hashes"] = {
        name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("cross_validation.py", "models.py")
    }
    if progress is not None:
        progress.clear()
        progress.update(result)
        result = progress
    for seed in cfg.training_seeds:
        for split in splits:
            fold = split["fold"]
            actual_seed = (seed+1009*fold) % 2**32
            mean, std = fit_normalizer(records, set(split["train"]))
            train_ds = WindowDataset(records, windows, set(split["train"]), mean, std)
            validation_ds = WindowDataset(records, windows, set(split["validation"]), mean, std)
            print(f"Seed {seed}, outer fold {fold}/{cfg.outer_folds}: "
                  f"inner train={len(split['train'])}, inner validation={len(split['validation'])}, "
                  f"outer test={len(split['test'])} participants", flush=True)
            chosen_state, chosen_index, best_loss = None, None, float("inf")
            trials = []
            for index, candidate in enumerate(candidates):
                state, trial = fit_candidate(train_ds, validation_ds, candidate, cfg,
                                             actual_seed, device, verbose=verbose)
                trials.append(trial)
                if trial["inner_metrics"]["log_loss"] < best_loss:
                    best_loss = trial["inner_metrics"]["log_loss"]
                    chosen_state, chosen_index = state, index
                del state
            # No refit: use the exact inner-selected checkpoint and normalization.
            model = build_model(**ARCHITECTURE).to(device)
            model.load_state_dict(chosen_state)
            test_ds = WindowDataset(records, windows, set(split["test"]), mean, std)
            test_windows = predict(model, test_ds, cfg.batch_size, device, cfg.num_workers)
            participants = aggregate_predictions(test_windows)
            if {r["subject"] for r in participants} != set(split["test"]):
                raise AssertionError("Test participant coverage mismatch")
            result["runs"].append(dict(seed=seed, fold=fold, actual_training_seed=actual_seed,
                                      selected_candidate=chosen_index, trials=trials,
                                      normalizer=dict(mean=mean.tolist(), std=std.tolist()),
                                      test_windows=test_windows, test_participants=participants,
                                      participant_metrics=metrics(participants), window_metrics=metrics(test_windows)))
            del model, chosen_state, train_ds, validation_ds, test_ds
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result["summaries"] = {}
    for seed in cfg.training_seeds:
        runs = [r for r in result["runs"] if r["seed"] == seed]
        participants = [p for r in runs for p in r["test_participants"]]
        if len(participants) != len(eligible) or {p["subject"] for p in participants} != eligible:
            raise AssertionError("OOF predictions must cover each eligible participant once per seed")
        result["summaries"][str(seed)] = dict(
            participant_metrics=metrics(participants),
            window_metrics=metrics([p for r in runs for p in r["test_windows"]]),
            confidence_intervals=confidence_intervals(participants, cfg.bootstrap_samples, cfg.bootstrap_seed),
            fold_accuracy_std=float(np.std([r["participant_metrics"]["accuracy"] for r in runs], ddof=1)))
    result["complete"] = True
    return result


def export_results(result, destination):
    """Explicit opt-in export; never overwrite existing artifacts or export raw signals."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    payload = json.dumps(result, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("results.json", payload)
    return path