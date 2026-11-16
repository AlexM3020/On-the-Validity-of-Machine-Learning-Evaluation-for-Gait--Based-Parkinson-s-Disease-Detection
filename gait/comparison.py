"""Equal-budget inner random search for Mamba/CNN on matched outer folds.

Clinical attributes are deliberately NOT accepted by the training API.
"""

from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import cross_validation as cv
from .comparison_models import build_candidate


def sample_candidates(n=8, search_seed=90210):
    """One fixed reference + n-1 unique samples/family, generated before training."""
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError("Use 1..100 candidates per family")
    candidates = {}
    for family in ("mamba", "cnn"):
        rng = np.random.default_rng(search_seed)  # matched shared choices by trial index
        base = dict(name=f"{family}_00", family=family, width=64 if family == "mamba" else 32,
                    layers=2 if family == "mamba" else 4, state=16 if family == "mamba" else None,
                    kernel=16 if family == "mamba" else 7, dropout=.2, pooling="mean_std",
                    seq_len=1000, normalization="training", lr=1e-4, weight_decay=1e-3)
        trials = [base]
        while len(trials) < n:
            c = dict(base, name=f"{family}_{len(trials):02d}")
            c.update(seq_len=int(rng.choice([500, 1000, 2000])),
                     normalization=str(rng.choice(["training", "recording"])),
                     dropout=float(rng.choice([.0, .1, .2, .3])),
                     pooling=str(rng.choice(["max", "mean_std"])),
                     lr=float(10**rng.uniform(np.log10(2.5e-5), np.log10(3e-4))),
                     weight_decay=float(10**rng.uniform(-4, -2)))
            c.update(width=int(rng.choice([32, 64, 128] if family == "mamba" else [32, 48, 64])),
                     layers=int(rng.choice([1, 2] if family == "mamba" else [3, 4])),
                     kernel=int(rng.choice([8, 16, 32] if family == "mamba" else [5, 7])))
            state_choice = int(rng.choice([16, 32]))  # advance both family streams identically
            c["state"] = state_choice if family == "mamba" else None
            trials.append(c)
        candidates[family] = trials
    return candidates


def validate_search(cfg, candidates):
    # Reuse existing protocol guards without treating old optimizer candidates as this schema.
    cv.validate_protocol(cfg, cv.DEFAULT_CANDIDATES)
    if cfg.overlap != 0:
        raise ValueError("Comparison fixes nonoverlap; change the registered protocol for overlap experiments")
    if set(candidates) != {"mamba", "cnn"} or not candidates["mamba"] or len(candidates["mamba"]) != len(candidates["cnn"]):
        raise ValueError("Both families require the same nonzero candidate budget")
    expected = {"name", "family", "width", "layers", "state", "kernel", "dropout", "pooling",
                "seq_len", "normalization", "lr", "weight_decay"}
    for family, trials in candidates.items():
        if len({t["name"] for t in trials}) != len(trials):
            raise ValueError("Duplicate candidate names")
        for c in trials:
            if set(c) != expected or c["family"] != family:
                raise ValueError("Invalid candidate schema")
            for name in ("width", "layers", "kernel", "seq_len"):
                if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < 1:
                    raise ValueError(f"Invalid {name}")
            if c["pooling"] not in ("max", "mean_std") or c["normalization"] not in ("training", "recording"):
                raise ValueError("Invalid pooling or normalization")
            if not 0 <= c["dropout"] < 1 or not cfg.min_lr <= c["lr"] or not 0 <= c["weight_decay"]:
                raise ValueError("Invalid numeric candidate")
            if not np.isfinite([c["dropout"], c["lr"], c["weight_decay"]]).all():
                raise ValueError("Nonfinite candidate")
            if family == "mamba" and c["state"] not in (16, 32):
                raise ValueError("Mamba state must be 16 or 32")
            if family == "cnn" and (c["state"] is not None or c["kernel"] % 2 != 1):
                raise ValueError("CNN requires odd kernel and no SSM state")


def prepare_datasets(records, split, candidate):
    """Deterministic transforms; no clinical metadata, test-label selection, or augmentation."""
    if candidate["normalization"] == "training":
        mean, std = cv.fit_normalizer(records, set(split["train"]))
        transformed = records
        stats = dict(method="training", mean=mean.tolist(), std=std.tolist())
    else:
        transformed, per_record = [], {}
        # Offline recording-wise standardization uses only each recording itself.
        for r in records:
            x = r["signal"].astype(np.float64)
            mu, sigma = x.mean(0), x.std(0)
            sigma[sigma < 1e-8] = 1
            transformed.append({**r, "signal": ((x-mu)/sigma).astype(np.float32)})
            per_record[r["recording"]] = dict(mean=mu.tolist(), std=sigma.tolist())
        mean, std = np.zeros(18, dtype=np.float32), np.ones(18, dtype=np.float32)
        stats = dict(method="recording", recordings=per_record)
    windows = cv.make_windows(transformed, candidate["seq_len"], 0)
    datasets = {part: cv.WindowDataset(transformed, windows, set(split[part]), mean, std)
                for part in ("train", "validation", "test")}
    for part, ds in datasets.items():
        if {w["subject"] for w in ds.windows} != set(split[part]):
            raise ValueError("Candidate lost participants: use a common minimum-length cohort")
    return datasets, stats


def fit_trial(training, validation, c, cfg, seed, device, verbose=False):
    cv.seed_everything(seed)
    model = build_candidate(c).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    receptive_field = getattr(model, "receptive_field", None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.scheduler_factor,
                    patience=cfg.scheduler_patience, min_lr=cfg.min_lr)
    weights = cv.class_weights(training).to(device)
    loader = DataLoader(training, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed), num_workers=cfg.num_workers)
    best, state, stale, history, best_epoch = float("inf"), None, 0, [], 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        numerator = denominator = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            summed = nn.functional.cross_entropy(model(x), y, weight=weights, reduction="sum")
            mass = weights[y].sum()
            loss = summed/mass
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            numerator += summed.item()
            denominator += mass.item()
        rows = cv.aggregate_predictions(cv.predict(model, validation, cfg.batch_size, device, cfg.num_workers))
        scores = cv.metrics(rows)
        history.append(dict(epoch=epoch, train_loss=numerator/denominator,
                            validation_log_loss=scores["log_loss"],
                            validation_balanced_accuracy=scores["balanced_accuracy"],
                            lr=optimizer.param_groups[0]["lr"]))
        scheduler.step(scores["log_loss"])
        if scores["log_loss"] < best-1e-6:
            best, stale, best_epoch = scores["log_loss"], 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_rows, best_scores = rows, scores
        else:
            stale += 1
        if verbose:
            print(f"  {c['name']} epoch {epoch}: inner NLL={scores['log_loss']:.4f}, stale={stale}", flush=True)
        if stale >= cfg.patience:
            break
    if state is None:
        raise RuntimeError("No valid checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trial = dict(candidate=copy.deepcopy(c), inner_metrics=best_scores, best_epoch=best_epoch,
                 validation_participants=best_rows, history=history, parameters=parameter_count,
                 receptive_field_samples=receptive_field, seconds=time.perf_counter()-start,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                 class_weights=weights.cpu().tolist())
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, trial


def source_hashes():
    return {n: hashlib.sha256((Path(__file__).parent/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in ("models.py", "cross_validation.py", "comparison_models.py", "comparison.py",
                      "official_data.py", "diagnostics.py")}


def finish_summaries(result):
    cfg = result["protocol"]
    expected = {p for s in result["splits"] for p in s["test"]}
    summaries = {}
    for family in result["candidates"]:
        summaries[family] = {}
        for seed in cfg["training_seeds"]:
            runs = [r for r in result["runs"] if (r["family"], r["seed"]) == (family, seed)]
            if sorted(r["fold"] for r in runs) != list(range(1, cfg["outer_folds"]+1)):
                raise ValueError("Missing or repeated outer folds")
            rows = [p for r in runs for p in r["test_participants"]]
            if len(rows) != len(expected) or {p["subject"] for p in rows} != expected:
                raise ValueError("OOF participant coverage is not exactly once")
            summaries[family][str(seed)] = dict(
                participant_metrics=cv.metrics(rows),
                confidence_intervals=cv.confidence_intervals(rows, cfg["bootstrap_samples"], cfg["bootstrap_seed"]),
                window_metrics=cv.metrics([p for r in runs for p in r["test_windows"]]),
                fold_accuracy_std=float(np.std([r["participant_metrics"]["accuracy"] for r in runs], ddof=1)))
    result["summaries"] = summaries
    result["complete"] = True


def run_comparison(records, audit, provenance, cfg, candidates, *, progress=None, max_new_tasks=None, verbose=False):
    """Resumable between completed family/seed/fold tasks; no automatic disk writes.

    A task selects every candidate internally then scores its outer test once.
    Interrupted tasks restart. Never skip failed candidates or silently reduce a budget.
    """
    validate_search(cfg, candidates)
    if max_new_tasks is not None and (not isinstance(max_new_tasks, int) or max_new_tasks < 1):
        raise ValueError("max_new_tasks must be a positive integer or None")
    max_length = max(c["seq_len"] for ts in candidates.values() for c in ts)
    if any(len(r["signal"]) < max_length for r in records):
        raise ValueError("Use a common cohort eligible for the largest candidate window")
    # Detect accidentally modified RAM arrays, not just trusting stored force hashes.
    signal_hashes = {r["recording"]: hashlib.sha256(r["signal"].tobytes()).hexdigest() for r in records}
    if len(signal_hashes) != len(records) or len(set(signal_hashes.values())) != len(records):
        raise ValueError("Duplicate recording names/signals remain")
    splits = cv.make_splits(records, cfg)
    plan = dict(protocol=asdict(cfg), candidates=candidates, audit=audit, provenance=provenance,
                splits=splits, signal_hashes=signal_hashes, source_hashes=source_hashes())
    # JSON canonicalization permits exact comparisons with exported/reloaded bundles.
    plan = json.loads(json.dumps(plan, sort_keys=True, allow_nan=False))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    result = {} if progress is None else progress
    if result:
        if result.get("plan_signature") != signature or any(result.get(k) != v for k, v in plan.items()):
            raise ValueError("Resume rejected: configuration, cohort, splits or implementation changed")
        if result.get("environment") != cv.runtime_info():
            raise ValueError("Resume rejected: runtime environment changed")
    else:
        result.update(**plan, plan_signature=signature, environment=cv.runtime_info(), runs=[], complete=False)
    tasks = [(family, seed, s) for seed in cfg.training_seeds for s in splits for family in candidates]
    keys = [(r["family"], r["seed"], r["fold"]) for r in result["runs"]]
    if keys != [(f, seed, s["fold"]) for f, seed, s in tasks[:len(keys)]]:
        raise ValueError("Resume tasks must form a unique consecutive prefix")
    for r in result["runs"]:
        split = splits[r["fold"]-1]
        if {p["subject"] for p in r["test_participants"]} != set(split["test"]) or len(r["test_participants"]) != len(split["test"]):
            raise ValueError("Resume test participant coverage mismatch")
        if len(r["trials"]) != len(candidates[r["family"]]):
            raise ValueError("Resume candidate budget mismatch")
        for trial, candidate in zip(r["trials"], candidates[r["family"]]):
            if trial["candidate"] != candidate:
                raise ValueError("Resume candidate contents mismatch")
            if not trial["history"] or trial["best_epoch"] not in [h["epoch"] for h in trial["history"]]:
                raise ValueError("Resume trial history invalid")
            if {p["subject"] for p in trial["validation_participants"]} != set(split["validation"]):
                raise ValueError("Resume inner participant mismatch")
        chosen = int(np.argmin([t["inner_metrics"]["log_loss"] for t in r["trials"]]))
        if r["selected_candidate"] != chosen:
            raise ValueError("Resume selection is inconsistent with inner scores")
        aggregate = cv.aggregate_predictions(r["test_windows"])
        if aggregate != r["test_participants"] or cv.metrics(aggregate) != r["participant_metrics"]:
            raise ValueError("Resume stored outer predictions/metrics disagree")
    if result.get("complete"):
        finish_summaries(result)
        return result
    device = cv.require_cuda()
    for offset, (family, seed, split) in enumerate(tasks[len(keys):]):
        if max_new_tasks is not None and offset >= max_new_tasks:
            return result
        actual_seed = (seed+1009*split["fold"]) % 2**32
        print(f"Task {len(result['runs'])+1}/{len(tasks)}: {family}, seed {seed}, fold {split['fold']}/{cfg.outer_folds}", flush=True)
        best, state, selected, trials = float("inf"), None, None, []
        for index, c in enumerate(candidates[family]):
            datasets, _ = prepare_datasets(records, split, c)
            candidate_state, trial = fit_trial(datasets["train"], datasets["validation"], c, cfg, actual_seed, device, verbose)
            trials.append(trial)
            if trial["inner_metrics"]["log_loss"] < best:
                best, state, selected = trial["inner_metrics"]["log_loss"], candidate_state, index
            print(f"  {c['name']}: best inner NLL={trial['inner_metrics']['log_loss']:.4f}, epoch={trial['best_epoch']}, seconds={trial['seconds']:.0f}", flush=True)
            del candidate_state, datasets
        c = candidates[family][selected]
        datasets, normalizer = prepare_datasets(records, split, c)
        model = build_candidate(c).to(device)
        model.load_state_dict(state)
        test_windows = cv.predict(model, datasets["test"], cfg.batch_size, device, cfg.num_workers)
        participants = cv.aggregate_predictions(test_windows)
        if len(participants) != len(split["test"]) or {p["subject"] for p in participants} != set(split["test"]):
            raise ValueError("Outer prediction coverage mismatch")
        result["runs"].append(dict(family=family, seed=seed, fold=split["fold"], actual_training_seed=actual_seed,
                                  selected_candidate=selected, trials=trials, normalizer=normalizer,
                                  test_windows=test_windows, test_participants=participants,
                                  participant_metrics=cv.metrics(participants), window_metrics=cv.metrics(test_windows)))
        del model, state, datasets
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finish_summaries(result)
    return result


def smoke_candidates(candidates, batch_size=2):
    """Actual forward/backward for every proposed architecture before long runs."""
    device = cv.require_cuda()
    rows = []
    for family, trials in candidates.items():
        for c in trials:
            model = build_candidate(c).to(device)
            x = torch.randn(batch_size, c["seq_len"], 18, device=device)
            out = model(x)
            if out.shape != (batch_size, 2) or not torch.isfinite(out).all():
                raise ValueError("Invalid smoke-check logits")
            nn.functional.cross_entropy(out, torch.arange(batch_size, device=device) % 2).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Invalid smoke-check gradients")
            rows.append(dict(family=family, candidate=c["name"], parameters=sum(p.numel() for p in model.parameters()),
                             receptive_field_samples=getattr(model, "receptive_field", None)))
            del model, x, out
            torch.cuda.empty_cache()
    return rows