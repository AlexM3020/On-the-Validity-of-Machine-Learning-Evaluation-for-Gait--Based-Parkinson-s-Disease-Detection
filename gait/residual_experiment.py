"""Fold-fitted LPR extensions, not an original-LPGNet reproduction.

Two-stage training: window encoder, then participant-level head calibration.
Only original inner-training controls fit the prediction reference. No Mamba
dependency, downloads or file writes occur on import or training.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from . import cross_validation as cv
from .diagnostics import participant_errors, subgroup_table, seed_stability


VARIANTS = {"lpr11_stats": (11,), "multiscale_stats": (5, 11, 21)}
WARMUP = 22  # Common support: largest order + moving-average startup sample.


@dataclass(frozen=True)
class ResidualProtocol:
    outer_folds: int = 10
    inner_fraction: float = .2
    split_seed: int = 42
    training_seeds: tuple = (42, 123, 2026)
    window_samples: int = 100  # 2 seconds at 50 Hz; no overlap by default.
    batch_size: int = 64
    encoder_epochs: int = 60
    head_epochs: int = 40
    patience: int = 10
    scheduler_patience: int = 4
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 7301
    head_lr: float = 1e-3
    ridge: float = 1e-4


def candidate_search(count=3, seed=9011):
    """Identical optimizer draws across both extensions, frozen before evaluation."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
        raise ValueError("Use 1..20 candidates per extension")
    rng = np.random.default_rng(seed)
    rows = [dict(name="reference", lr=5e-4, weight_decay=1e-3, dropout=.25)]
    for i in range(1, count):
        rows.append(dict(name=f"random_{i:02d}", lr=float(10**rng.uniform(-4, -3)),
                         weight_decay=float(10**rng.uniform(-4, -2)),
                         dropout=float(rng.choice([.1, .2, .25, .3]))))
    return rows


def validate(cfg, candidates):
    for name in ("outer_folds", "split_seed", "window_samples", "batch_size", "encoder_epochs",
                 "head_epochs", "patience", "scheduler_patience", "bootstrap_samples", "bootstrap_seed"):
        v = getattr(cfg, name)
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if "seed" in name else 1):
            raise ValueError(f"Invalid {name}")
    if cfg.outer_folds < 2 or cfg.window_samples < 32 or not 0 < cfg.inner_fraction < 1:
        raise ValueError("Invalid fold, window or validation configuration")
    if not cfg.training_seeds or len(set(cfg.training_seeds)) != len(cfg.training_seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    for seed in (*cfg.training_seeds, cfg.split_seed, cfg.bootstrap_seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Invalid seed")
    if not np.isfinite([cfg.ridge, cfg.head_lr]).all() or cfg.ridge <= 0 or cfg.head_lr <= 0:
        raise ValueError("Ridge and head learning rate must be finite and positive")
    if not candidates or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique and nonempty")
    for c in candidates:
        if set(c) != {"name", "lr", "weight_decay", "dropout"}:
            raise ValueError("Unexpected candidate fields")
        if not np.isfinite([c["lr"], c["weight_decay"], c["dropout"]]).all() or c["lr"] <= 0 or c["weight_decay"] < 0 or not 0 <= c["dropout"] < 1:
            raise ValueError("Invalid candidate hyperparameters")


def downsample(records):
    """Causal two-point moving average followed by decimation; fixed, label-free."""
    output = []
    for r in records:
        x = np.asarray(r["signal"], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 18 or not np.isfinite(x).all():
            raise ValueError("Expected finite 18-channel original signal")
        output.append({**r, "signal": lfilter([.5, .5], [1.0], x, axis=0)[::2]})
    return output


def fit_reference(records50, train_subjects, ridge=1e-4):
    """Participant-balanced, recording-balanced biased autocorrelation LPC.

    Estimate variance scaling from controls only; divide by std without centering
    to retain LPGNet's scaling convention. Solve ridge Yule-Walker independently
    for each channel and order. Each recording starts a fresh history; no
    concatenation transitions enter autocorrelations. No test-specific LP refit.
    """
    by_subject = defaultdict(list)
    for r in records50:
        if r["subject"] in train_subjects and r["label"] == 0:
            if len(r["signal"]) <= WARMUP:
                raise ValueError("Control recording too short for reference")
            by_subject[r["subject"]].append(r)
    if not by_subject:
        raise ValueError("Reference requires inner-training controls")
    means, seconds = [], []
    for group in by_subject.values():
        means.append(np.mean([r["signal"].mean(0) for r in group], axis=0))
        seconds.append(np.mean([(r["signal"]**2).mean(0) for r in group], axis=0))
    mean = np.mean(means, axis=0)
    variance = np.maximum(np.mean(seconds, axis=0)-mean**2, 0)
    scale = np.sqrt(variance)
    constant = scale < 1e-8
    scale[constant] = 1
    participant_corr = []
    for group in by_subject.values():
        recording_corr = []
        for r in group:
            x = r["signal"]/scale
            recording_corr.append(np.stack([(x[k:]*x[:len(x)-k]).sum(0)/len(x) for k in range(22)]))
        participant_corr.append(np.mean(recording_corr, axis=0))
    corr = np.mean(participant_corr, axis=0)
    coefficients = {}
    for order in (5, 11, 21):
        channel_filters = []
        for c in range(18):
            if corr[0, c] < 1e-12:
                a = np.r_[1., np.zeros(order)]
            else:
                col = corr[:order, c].copy()
                col[0] += ridge*corr[0, c]
                a = np.r_[1., solve_toeplitz(col, -corr[1:order+1, c])]
                if not np.isfinite(a).all() or np.max(np.abs(np.roots(a))) >= 1:
                    raise ValueError("Nonfinite or unstable reference predictor")
            channel_filters.append(a.tolist())
        coefficients[str(order)] = channel_filters
    return dict(scale=scale.tolist(), coefficients=coefficients, ridge=ridge, warmup=WARMUP,
                training_control_subjects=sorted(by_subject),
                training_control_recordings=sorted(r["recording"] for group in by_subject.values() for r in group),
                constant_channels=np.flatnonzero(constant).tolist(),
                estimator="equal-participant/equal-recording biased autocorrelation ridge Yule-Walker",
                normalization="inner-training-control standard deviation; no centering")


def transform(records50, reference, orders):
    """Sensor-major residual bank [T,18,orders], flattened for grouped fusion."""
    scale = np.array(reference["scale"])
    output = []
    for r in records50:
        x = r["signal"]/scale
        bank = np.stack([np.column_stack([lfilter(reference["coefficients"][str(p)][c], [1.], x[:, c])
                                         for c in range(18)]) for p in orders], axis=-1)
        bank = bank[WARMUP:].reshape(-1, 18*len(orders)).astype(np.float32)
        if not np.isfinite(bank).all():
            raise ValueError("Nonfinite residuals")
        output.append({**r, "signal": bank})
    return output


class ResidualWindows(Dataset):
    def __init__(self, records, subjects, length):
        self.records = [r for r in records if r["subject"] in subjects]
        self.windows = [(i, start) for i, r in enumerate(self.records)
                        for start in range(0, len(r["signal"])-length+1, length)]
        self.labels = [self.records[i]["label"] for i, _ in self.windows]
        self.length = length
        if {self.records[i]["subject"] for i, _ in self.windows} != set(subjects):
            raise ValueError("Every training participant must have at least one complete window")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        i, start = self.windows[index]
        return torch.from_numpy(self.records[i]["signal"][start:start+self.length]), self.labels[index]


class SeparableBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel, pool):
        super().__init__()
        self.depthwise = nn.Conv1d(incoming, incoming, kernel, groups=incoming, bias=False)
        self.pointwise = nn.Conv1d(incoming, outgoing, 1)
        self.norm = nn.LayerNorm(outgoing)  # Across channels only; supports variable recording lengths.
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        x = self.activation(self.norm(x.transpose(1, 2))).transpose(1, 2)
        return self.pool(x)


class ResidualStatsCNN(nn.Module):
    def __init__(self, n_orders, dropout=.25):
        super().__init__()
        # Both arms produce exactly 2 features/sensor; bank changes only fusion input width.
        self.fusion = nn.Conv1d(18*n_orders, 36, 1, groups=18)
        self.blocks = nn.Sequential(SeparableBlock(36, 32, 7, True),
                                    SeparableBlock(32, 32, 5, True),
                                    SeparableBlock(32, 64, 3, False))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))
        self.receptive_field_samples = 26  # Residual input support before statistics pooling.

    def encode(self, x):
        features = self.blocks(self.fusion(x.transpose(1, 2)))
        return torch.cat([features.mean(-1), features.var(-1, unbiased=False).clamp_min(1e-8).sqrt()], dim=1)

    def forward(self, x):
        return self.head(self.encode(x))


@torch.no_grad()
def participant_embeddings(model, records, subjects, device):
    """Full recording statistics, then equal recording mean per participant.

    No padding, window-probability averaging, or cross-record filter histories.
    Features are pooled before the linear head. Duration is not used as a weight.
    """
    model.eval()
    groups = defaultdict(list)
    for r in records:
        if r["subject"] in subjects:
            embedding = model.encode(torch.from_numpy(r["signal"])[None].to(device)).cpu()[0]
            groups[r["subject"]].append((embedding, r))
    if set(groups) != set(subjects):
        raise ValueError("Participant embedding coverage mismatch")
    embeddings, metadata = [], []
    for subject, group in sorted(groups.items()):
        labels = {r["label"] for _, r in group}
        if len(labels) != 1:
            raise ValueError("Conflicting participant labels")
        embeddings.append(torch.stack([e for e, _ in group]).mean(0))
        metadata.append(dict(subject=subject, label=next(iter(labels)), recordings=len(group),
                             windows=0, scoring_unit="participant_embedding",
                             recording_ids=[r["recording"] for _, r in group]))
    return torch.stack(embeddings), metadata


@torch.no_grad()
def score_head(head, embeddings, metadata, device):
    head.eval()
    p = head(embeddings.to(device)).softmax(-1)[:, 1].cpu().tolist()
    rows = [{**row, "probability":float(prob), "prediction":int(prob >= .5)} for row, prob in zip(metadata, p)]
    return rows, cv.metrics(rows)


def train_trial(training_records, validation_records, candidate, cfg, orders, seed, device, verbose=False):
    """No test data argument: both stages select only on inner validation NLL."""
    cv.seed_everything(seed)
    model = ResidualStatsCNN(len(orders), candidate["dropout"]).to(device)
    train_subjects = {r["subject"] for r in training_records}
    val_subjects = {r["subject"] for r in validation_records}
    if train_subjects & val_subjects:
        raise ValueError("Inner participant overlap")
    windows = ResidualWindows(training_records, train_subjects, cfg.window_samples)
    counts = np.bincount(windows.labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Training requires both classes")
    weights = torch.tensor(counts.sum()/(2*counts), dtype=torch.float32, device=device)
    loader = DataLoader(windows, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["lr"], weight_decay=candidate["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.scheduler_patience, factor=.5)
    best, stale, best_state, history, best_epoch = float("inf"), 0, None, [], 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.encoder_epochs+1):
        model.train()
        numerator = denominator = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            summed = nn.functional.cross_entropy(model(x), y, weight=weights, reduction="sum")
            mass = weights[y].sum()
            loss = summed/mass
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite encoder loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            numerator += summed.item()
            denominator += mass.item()
        emb, meta = participant_embeddings(model, validation_records, val_subjects, device)
        _, scores = score_head(model.head, emb, meta, device)
        history.append(dict(epoch=epoch, train_loss=numerator/denominator,
                            validation_log_loss=scores["log_loss"], lr=optimizer.param_groups[0]["lr"]))
        scheduler.step(scores["log_loss"])
        if scores["log_loss"] < best-1e-6:
            best, stale, best_epoch = scores["log_loss"], 0, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if verbose:
            print(f"    encoder {epoch}: inner NLL={scores['log_loss']:.4f}", flush=True)
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid encoder checkpoint")
    model.load_state_dict(best_state)
    # Stage 2: fixed recording embeddings and exactly one supervised sample/person.
    train_emb, train_meta = participant_embeddings(model, training_records, train_subjects, device)
    val_emb, val_meta = participant_embeddings(model, validation_records, val_subjects, device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    y_train = torch.tensor([r["label"] for r in train_meta], dtype=torch.long)
    head_loader = DataLoader(TensorDataset(train_emb, y_train), batch_size=cfg.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(seed+1))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.head_lr, weight_decay=candidate["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.scheduler_patience, factor=.5)
    # Epoch zero is a legitimate candidate: retain the encoder-stage head if calibration hurts.
    best_rows, best_scores = score_head(model.head, val_emb, val_meta, device)
    head_best, head_epoch, stale, head_history = best_scores["log_loss"], 0, 0, []
    for epoch in range(1, cfg.head_epochs+1):
        model.head.train()
        total, n = 0., 0
        for emb, y in head_loader:
            emb, y = emb.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model.head(emb), y)  # Unweighted participant objective.
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite calibration loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(y)
            n += len(y)
        rows, scores = score_head(model.head, val_emb, val_meta, device)
        head_history.append(dict(epoch=epoch, train_loss=total/n, validation_log_loss=scores["log_loss"],
                                 lr=optimizer.param_groups[0]["lr"]))
        scheduler.step(scores["log_loss"])
        if scores["log_loss"] < head_best-1e-6:
            head_best, head_epoch, stale = scores["log_loss"], epoch, 0
            best_rows, best_scores = rows, scores
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trial = dict(candidate=copy.deepcopy(candidate), orders=list(orders), parameters=sum(p.numel() for p in model.parameters()),
                 best_encoder_epoch=best_epoch, best_head_epoch=head_epoch,
                 history=history, head_history=head_history, inner_metrics=best_scores,
                 validation_participants=best_rows, window_class_weights=weights.cpu().tolist(),
                 seconds=time.perf_counter()-start,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, trial


def engine_hashes():
    return {n:hashlib.sha256((Path(__file__).parent/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in ("residual_experiment.py", "official_data.py", "cross_validation.py", "diagnostics.py")}


def runtime():
    import scipy
    return {**cv.runtime_info(), "scipy":scipy.__version__, "scoring":"mean full-recording embeddings per participant"}


def summarize(result):
    expected = {p for s in result["splits"] for p in s["test"]}
    summaries = {}
    for family in result["variants"]:
        summaries[family] = {}
        for seed in result["protocol"]["training_seeds"]:
            runs = [r for r in result["runs"] if (r["family"], r["seed"]) == (family, seed)]
            if sorted(r["fold"] for r in runs) != list(range(1, result["protocol"]["outer_folds"]+1)):
                raise ValueError("Incomplete outer folds")
            rows = [p for r in runs for p in r["test_participants"]]
            if len(rows) != len(expected) or {r["subject"] for r in rows} != expected:
                raise ValueError("OOF coverage must be exactly once per participant")
            summaries[family][str(seed)] = dict(participant_metrics=cv.metrics(rows),
                confidence_intervals=cv.confidence_intervals(rows, result["protocol"]["bootstrap_samples"], result["protocol"]["bootstrap_seed"]),
                fold_accuracy_std=float(np.std([r["participant_metrics"]["accuracy"] for r in runs], ddof=1)))
    result["summaries"], result["complete"] = summaries, True


def run(records, audit, provenance, cfg, candidates, *, progress=None, max_new_tasks=2, verbose=False, device="cuda"):
    validate(cfg, candidates)
    if device not in ("cuda", "cpu"):
        raise ValueError("Choose real PyTorch CPU or CUDA execution")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select CPU explicitly or enable Kaggle GPU")
    if max_new_tasks is not None and (not isinstance(max_new_tasks, int) or max_new_tasks < 1):
        raise ValueError("max_new_tasks must be positive or None")
    device = torch.device(device)
    hashes = {r["recording"]:hashlib.sha256(r["signal"].tobytes()).hexdigest() for r in records}
    if len(hashes) != len(records) or len(set(hashes.values())) != len(records):
        raise ValueError("Duplicate original records remain")
    records50 = downsample(records)
    if any(len(r["signal"])-WARMUP < cfg.window_samples for r in records50):
        raise ValueError("Common cohort contains recordings shorter than the training window")
    splits = cv.make_splits(records, cfg)
    plan = json.loads(json.dumps(dict(protocol=asdict(cfg), candidates=candidates, variants=VARIANTS,
                    splits=splits, audit=audit, provenance=provenance, signal_hashes=hashes,
                    source_hashes=engine_hashes(), environment=runtime(), device=str(device)), sort_keys=True, allow_nan=False))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    result = {} if progress is None else progress
    if result:
        if result.get("plan_signature") != signature or any(result.get(k) != v for k,v in plan.items()):
            raise ValueError("Resume rejected: plan, data, source or environment changed")
    else:
        result.update(**plan, plan_signature=signature, runs=[], complete=False)
    tasks = [(f, seed, split) for seed in cfg.training_seeds for split in splits for f in VARIANTS]
    keys = [(r["family"], r["seed"], r["fold"]) for r in result["runs"]]
    if keys != [(f, s, sp["fold"]) for f,s,sp in tasks[:len(keys)]]:
        raise ValueError("Completed tasks must be a unique consecutive prefix")
    checked_references = {}
    for completed in result["runs"]:
        split = splits[completed["fold"]-1]
        if completed["fold"] not in checked_references:
            checked_references[completed["fold"]] = fit_reference(records50, set(split["train"]), cfg.ridge)
        if completed["reference"] != checked_references[completed["fold"]]:
            raise ValueError("Resume fitted reference differs from inner-training controls")
        rows = completed["test_participants"]
        if len(rows) != len(split["test"]) or {p["subject"] for p in rows} != set(split["test"]):
            raise ValueError("Resume participant coverage mismatch")
        trials = completed["trials"]
        if len(trials) != len(candidates) or [t["candidate"] for t in trials] != candidates:
            raise ValueError("Resume candidate mismatch")
        if completed["selected_candidate"] != int(np.argmin([t["inner_metrics"]["log_loss"] for t in trials])):
            raise ValueError("Resume selection disagrees with inner scores")
        if cv.metrics(rows) != completed["participant_metrics"]:
            raise ValueError("Resume predictions and metrics disagree")
        for trial in trials:
            if trial["orders"] != list(VARIANTS[completed["family"]]):
                raise ValueError("Resume residual bank mismatch")
            vr = trial["validation_participants"]
            if len(vr) != len(split["validation"]) or {p["subject"] for p in vr} != set(split["validation"]):
                raise ValueError("Resume inner coverage mismatch")
            if cv.metrics(vr) != trial["inner_metrics"]:
                raise ValueError("Resume inner metrics mismatch")
    if result.get("complete"):
        summarize(result)
        return result
    for i, (family, seed, split) in enumerate(tasks[len(keys):]):
        if max_new_tasks is not None and i >= max_new_tasks:
            return result
        print(f"Task {len(result['runs'])+1}/{len(tasks)}: {family}, seed {seed}, outer fold {split['fold']}", flush=True)
        reference = fit_reference(records50, set(split["train"]), cfg.ridge)
        orders = VARIANTS[family]
        # No outer signals are transformed until all candidate choices are complete.
        inner = transform([r for r in records50 if r["subject"] in set(split["train"]+split["validation"])], reference, orders)
        training = [r for r in inner if r["subject"] in split["train"]]
        validation = [r for r in inner if r["subject"] in split["validation"]]
        trials, best, state, selected = [], float("inf"), None, None
        actual_seed = (seed+1009*split["fold"]) % 2**32
        for index, candidate in enumerate(candidates):
            candidate_state, trial = train_trial(training, validation, candidate, cfg, orders, actual_seed, device, verbose)
            trials.append(trial)
            if trial["inner_metrics"]["log_loss"] < best:
                best, state, selected = trial["inner_metrics"]["log_loss"], candidate_state, index
            print(f"  {candidate['name']}: inner NLL={trial['inner_metrics']['log_loss']:.4f}, encoder epoch={trial['best_encoder_epoch']}, head epoch={trial['best_head_epoch']}", flush=True)
            del candidate_state
        model = ResidualStatsCNN(len(orders), candidates[selected]["dropout"]).to(device)
        model.load_state_dict(state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        test = transform([r for r in records50 if r["subject"] in split["test"]], reference, orders)
        emb, meta = participant_embeddings(model, test, set(split["test"]), device)
        test_rows, scores = score_head(model.head, emb, meta, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["runs"].append(dict(family=family, seed=seed, fold=split["fold"], actual_training_seed=actual_seed,
                                  reference=reference, selected_candidate=selected, trials=trials,
                                  test_participants=test_rows, participant_metrics=scores,
                                  test_transform_encode_score_seconds=time.perf_counter()-start))
        del model, state, inner, training, validation, test
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summarize(result)
    return result


def reports(result, clinical):
    outer = participant_errors(result, clinical)
    inner = participant_errors(result, clinical, "validation")
    pd = [{**r,"hy_group":str(r["hy"]) if r["hy_status"] == "verified" and r["hy"] is not None else "Unknown"}
          for r in outer if r["label"]]
    severity = subgroup_table(pd, ["hy_group"])
    for family in result["variants"]:
        for seed in result["protocol"]["training_seeds"]:
            for stage in ("2.0", "2.5", "3.0", "Unknown"):
                if not any((r["family"],r["seed"],r["hy_group"]) == (family,seed,stage) for r in severity):
                    severity.append(dict(family=family,seed=seed,hy_group=stage,n=0,sensitivity=None))
    # Matched participant resampling for multi-order minus single-order metrics,
    # preserving each participant's repeated-seed block. Not a prediction ensemble.
    seeds = result["protocol"]["training_seeds"]
    subjects = sorted({r["subject"] for r in outer})
    index = {(r["family"],r["seed"],r["subject"]):r for r in outer}
    y = np.array([index[("lpr11_stats",seeds[0],s)]["label"] for s in subjects])
    groups = [np.flatnonzero(y == c) for c in (0,1)]
    names = ("accuracy","balanced_accuracy","macro_f1","sensitivity","specificity","roc_auc")
    def difference(ids):
        deltas = []
        for seed in seeds:
            scores = {f:cv.metrics([index[(f,seed,subjects[i])] for i in ids]) for f in VARIANTS}
            deltas.append([scores["multiscale_stats"][k]-scores["lpr11_stats"][k] for k in names])
        return np.mean(deltas, axis=0)
    point = difference(np.arange(len(subjects)))
    rng = np.random.default_rng(result["protocol"]["bootstrap_seed"])
    samples = np.array([difference(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]))
                        for _ in range(result["protocol"]["bootstrap_samples"])])
    comparison = [dict(metric=k,multiscale_minus_single=float(point[i]),
                       lower=float(np.percentile(samples[:,i],2.5)),upper=float(np.percentile(samples[:,i],97.5)))
                  for i,k in enumerate(names)]
    return dict(participant_errors=outer, by_study=subgroup_table(outer,["study"]),
                by_study_diagnosis=subgroup_table(outer,["study","diagnosis"]),
                by_fold=subgroup_table(outer,["fold"]), pd_severity=severity,
                metadata_coverage=subgroup_table(outer,["diagnosis","hy_status"]),
                confidently_wrong=sorted([r for r in outer if r["confident_error"]],key=lambda r:-r["nll"]),
                inner_high_loss=sorted(inner,key=lambda r:-r["nll"]),
                seed_stability=seed_stability(outer,seeds),paired_comparison=comparison)