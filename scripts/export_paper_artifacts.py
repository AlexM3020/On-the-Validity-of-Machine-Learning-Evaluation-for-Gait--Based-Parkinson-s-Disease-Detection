"""Export every table and figure in the evaluation-validity manuscript.

The inputs are the repository's immutable result archives, retained original LSTM
figures/reports, and loss histories recovered from verified factorial checkpoints.
PhysioNet access is limited to the checksum-verified manifest and clinical metadata;
no waveform download or model training is performed by this exporter.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gait import diagnostics  # noqa: E402
from gait.official_data import load_verified_clinical  # noqa: E402
from scripts.replay_partition_audit import replay_audit  # noqa: E402

DIRECT_ARCHIVE = ROOT / "notebooks/pd-mamba-cnn-results/comparison_progress_20261116T204946297832Z.zip"
FACTORIAL_ARCHIVE = ROOT / "notebooks/results_lpgnet_replication/reports.zip"
LSTM_ASSETS = ROOT / "paper_assets/lstm"
LPGNET_ASSETS = ROOT / "paper_assets/lpgnet"
CONDITIONS = ["global__test", "local__test", "global__inner", "local__inner"]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc")
ALL_METRICS = METRICS + ("log_loss", "brier")
N_BOOT = 1000
BOOTSTRAP_SEED = 2028


def metric_dict(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be nonempty, aligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Invalid binary labels or probabilities")
    prediction = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "brier": float(np.mean((p - y) ** 2)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def write_table(output: Path, name: str, rows) -> str:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    target = output / f"{name}.csv"
    frame.to_csv(target, index=False)
    return target.name


def read_factorial() -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(FACTORIAL_ARCHIVE) as archive:
        result = {}
        for name in ("summary", "fold_metrics", "participants", "participant_fold_metrics", "pd_severity"):
            result[name] = pd.read_csv(io.BytesIO(archive.read(f"reports/{name}.csv")))
        result["run_plan"] = json.loads(archive.read("reports/run_plan.json"))
    return result


def read_direct() -> dict:
    with zipfile.ZipFile(DIRECT_ARCHIVE) as archive:
        result = json.loads(archive.read("results.json"))
    if not result.get("complete") or len(result["runs"]) != 60:
        raise ValueError("The direct result archive is not complete")
    return result


def pooled_intervals(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in sorted(participants["condition"].unique()):
        group = participants[participants["condition"] == condition].sort_values("subject")
        if len(group) != 165 or group["subject"].duplicated().any():
            raise ValueError("Incomplete factorial participant inventory")
        y = group["label"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        classes = [np.flatnonzero(y == value) for value in (0, 1)]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for metric in METRICS:
            observed = metric_dict(y, p)[metric]
            values = []
            for _ in range(N_BOOT):
                indices = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in classes])
                values.append(metric_dict(y[indices], p[indices])[metric])
            rows.append({
                "condition": condition,
                "metric": metric,
                "estimate": observed,
                "bootstrap_95_low": np.percentile(values, 2.5),
                "bootstrap_95_high": np.percentile(values, 97.5),
                "bootstrap_samples": N_BOOT,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    return pd.DataFrame(rows)


def paired_contrasts(participants: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("global__test", "global__inner"),
        ("local__test", "local__inner"),
        ("global__test", "local__test"),
        ("global__inner", "local__inner"),
        ("global__test", "local__inner"),
    ]
    rows = []
    for first, second in comparisons:
        a = participants[participants["condition"] == first].set_index("subject").sort_index()
        b = participants[participants["condition"] == second].set_index("subject").sort_index()
        if not a.index.equals(b.index) or a.index.has_duplicates or b.index.has_duplicates:
            raise ValueError("Condition participant alignment differs")
        if not np.array_equal(a["label"], b["label"]) or not np.array_equal(a["fold"], b["fold"]):
            raise ValueError("Condition label/fold alignment differs")
        y = a["label"].to_numpy(dtype=int)
        pa = a["probability"].to_numpy(dtype=float)
        pb = b["probability"].to_numpy(dtype=float)
        classes = [np.flatnonzero(y == value) for value in (0, 1)]

        def delta(indices):
            am = metric_dict(y[indices], pa[indices])
            bm = metric_dict(y[indices], pb[indices])
            return np.array([am[name] - bm[name] for name in ALL_METRICS])

        point = delta(np.arange(len(y)))
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = np.array([
            delta(np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in classes]))
            for _ in range(N_BOOT)
        ])
        for i, metric in enumerate(ALL_METRICS):
            rows.append({
                "comparison": f"{first} minus {second}",
                "metric": metric,
                "difference": point[i],
                "bootstrap_95_low": np.percentile(samples[:, i], 2.5),
                "bootstrap_95_high": np.percentile(samples[:, i], 97.5),
                "participants": len(y),
                "paired_seeds": 1,
                "bootstrap_samples": N_BOOT,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    result = pd.DataFrame(rows)
    retained = pd.read_csv(LPGNET_ASSETS / "paired_comparisons.csv")
    columns = ["comparison", "metric", "difference", "bootstrap_95_low", "bootstrap_95_high"]
    check = result.merge(retained, on=["comparison", "metric"], suffixes=("_new", "_retained"), how="inner")
    if len(check) != len(result):
        raise AssertionError("Retained paired-contrast coverage differs")
    for column in columns[2:]:
        if not np.allclose(check[f"{column}_new"], check[f"{column}_retained"], rtol=0, atol=1e-12):
            raise AssertionError(f"Recomputed paired contrast differs: {column}")
    return result


def export_lpgnet(output: Path, factorial: dict) -> dict[str, list[str]]:
    summary = factorial["summary"].set_index("condition")
    participants = factorial["participants"]
    files: dict[str, list[str]] = {}

    rows = []
    for condition in CONDITIONS:
        row = summary.loc[condition]
        group = participants[participants["condition"] == condition]
        counts = metric_dict(group["label"], group["probability"])
        rows.append({
            "condition": condition,
            "accuracy_percent": 100 * row["participant_accuracy"],
            "balanced_accuracy_percent": 100 * row["participant_balanced_accuracy"],
            "sensitivity_percent": 100 * row["participant_sensitivity"],
            "specificity_percent": 100 * row["participant_specificity"],
            "roc_auc": row["participant_roc_auc"],
            "log_loss": row["participant_log_loss"],
            "brier": row["participant_brier"],
            "tn": counts["tn"], "fp": counts["fp"], "fn": counts["fn"], "tp": counts["tp"],
        })
    files["tab:lpgnet-factorial"] = [write_table(output, "tab-lpgnet-factorial", rows)]

    folds = factorial["fold_metrics"]
    released = folds[folds["condition"] == "global__test"]
    published = {"accuracy": (0.903, 0.058), "f1_pd": (0.932, 0.036), "roc_auc": (0.917, 0.094)}
    labels = {"accuracy": "Accuracy", "f1_pd": "PD F1", "roc_auc": "ROC AUC"}
    rows = []
    for metric in ("accuracy", "f1_pd", "roc_auc"):
        mean, sd = released[metric].mean(), released[metric].std(ddof=1)
        rows.append({"metric": labels[metric], "published_mean": published[metric][0],
                     "published_sd": published[metric][1], "reconstructed_mean": mean,
                     "reconstructed_sd": sd, "mean_difference": mean - published[metric][0]})
    files["tab:lpgnet-replication"] = [write_table(output, "tab-lpgnet-replication", rows)]

    protocol_rows = [
        ("global__test", "All dataset controls", "All outer-training / outer test", "Invalid: global LP and outer-test control"),
        ("local__test", "Controls in outer-training pool", "All outer-training / outer test", "Invalid: outer-test control"),
        ("global__inner", "All dataset controls", "Core training / inner validation", "Invalid: global LP includes outer-test controls"),
        ("local__inner", "Controls in core-training subset", "Core training / inner validation", "Two tested mechanisms corrected; not identity-clean"),
    ]
    files["tab:factorial-protocols"] = [write_table(output, "tab-factorial-protocols", [
        {"condition": c, "lp_reference_data": lp, "optimization_training_control": control, "validity": validity}
        for c, lp, control, validity in protocol_rows
    ])]

    intervals = pooled_intervals(participants)
    files["tab:lpgnet-factorial"].append(write_table(output, "tab-lpgnet-participant-intervals", intervals))
    contrasts = paired_contrasts(participants)
    files["tab:factorial-contrasts"] = [write_table(output, "tab-factorial-contrasts", contrasts)]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for condition in CONDITIONS:
        group = participants[participants["condition"] == condition]
        fpr, tpr, _ = roc_curve(group["label"], group["probability"])
        auc = summary.loc[condition, "participant_roc_auc"]
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{condition} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="0.55", linestyle=":", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    files["fig:lpgnet-roc"] = []
    for suffix in ("pdf", "png"):
        target = output / f"fig-lpgnet-roc.{suffix}"
        fig.savefig(target, dpi=220)
        files["fig:lpgnet-roc"].append(target.name)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax, condition in zip(axes.ravel(), CONDITIONS):
        group = participants[participants["condition"] == condition]
        counts = metric_dict(group["label"], group["probability"])
        matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > 50 else "black", fontsize=13)
        ax.set_title(condition)
        ax.set_xticks([0, 1], ["Control", "PD"])
        ax.set_yticks([0, 1], ["Control", "PD"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    files["fig:lpgnet-confusion"] = []
    for suffix in ("pdf", "png"):
        target = output / f"fig-lpgnet-confusion.{suffix}"
        fig.savefig(target, dpi=220)
        files["fig:lpgnet-confusion"].append(target.name)
    plt.close(fig)
    return files


def export_loss_figure(output: Path) -> list[str]:
    payload = json.loads((LPGNET_ASSETS / "recording_loss_histories.json").read_text())
    histories = payload["histories"]
    if payload["n_condition_fold_histories"] != 40 or len(histories) != 40:
        raise ValueError("Incomplete factorial history asset")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=False, sharey=True)
    for ax, condition in zip(axes.ravel(), CONDITIONS):
        selected = [h for h in histories if h["condition"] == condition]
        if len(selected) != 10 or sorted(h["fold"] for h in selected) != list(range(10)):
            raise ValueError(f"Incomplete loss histories for {condition}")
        max_epochs = max(len(h["loss"]) for h in selected)
        train, validation = [], []
        for epoch in range(max_epochs):
            train.append([h["loss"][epoch] for h in selected if len(h["loss"]) > epoch])
            validation.append([h["val_loss"][epoch] for h in selected if len(h["val_loss"]) > epoch])
        for history in selected:
            epochs = np.arange(1, len(history["loss"]) + 1)
            ax.plot(epochs, history["loss"], color="tab:blue", alpha=0.10, linewidth=0.7)
            ax.plot(epochs, history["val_loss"], color="tab:orange", alpha=0.10, linewidth=0.7)
        for values, color, linestyle, label in (
            (train, "tab:blue", "-", "Training median / IQR"),
            (validation, "tab:orange", "--", "Validation median / IQR"),
        ):
            median = np.array([np.median(v) for v in values])
            low = np.array([np.percentile(v, 25) for v in values])
            high = np.array([np.percentile(v, 75) for v in values])
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, median, color=color, linestyle=linestyle, linewidth=1.9, label=label)
            ax.fill_between(epochs, low, high, color=color, alpha=0.16, linewidth=0)
        ax.set_title(condition)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.18)
    axes[0, 0].set_ylabel("Label-smoothed binary-cross-entropy loss")
    axes[1, 0].set_ylabel("Label-smoothed binary-cross-entropy loss")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    files = []
    for suffix in ("pdf", "png"):
        target = output / f"fig-lpgnet-loss.{suffix}"
        fig.savefig(target, dpi=220)
        files.append(target.name)
    plt.close(fig)
    return files


def export_lstm(output: Path) -> dict[str, list[str]]:
    recording = pd.read_csv(LSTM_ASSETS / "classification_report_recording.csv", index_col=0)
    participant = pd.read_csv(LSTM_ASSETS / "classification_report_participant.csv", index_col=0)

    def row(metric, column):
        return {
            "metric": metric,
            "recording_level_split_percent": 100 * recording.loc[column, metric],
            "participant_disjoint_split_percent": 100 * participant.loc[column, metric],
        }

    rows = [
        row("precision", "accuracy") | {"metric": "Accuracy"},
        row("precision", "macro avg") | {"metric": "Macro precision"},
        row("recall", "macro avg") | {"metric": "Macro recall"},
        row("f1-score", "macro avg") | {"metric": "Macro F1"},
        row("f1-score", "Control") | {"metric": "Control F1"},
        row("f1-score", "Parkinson") | {"metric": "PD F1"},
    ]
    # The first row uses the classification report's explicit accuracy field in
    # all three metric columns; the remaining rows use their named class/average.
    rows[0] = {
        "metric": "Accuracy",
        "recording_level_split_percent": 100 * recording.loc["accuracy", "accuracy"] if "accuracy" in recording.columns else 100 * recording.loc["accuracy", "precision"],
        "participant_disjoint_split_percent": 100 * participant.loc["accuracy", "accuracy"] if "accuracy" in participant.columns else 100 * participant.loc["accuracy", "precision"],
    }
    expected = np.array([[92.50, 83.08], [92.62, 83.35], [92.83, 83.10],
                         [92.50, 83.05], [92.31, 83.74], [92.68, 82.35]])
    actual = np.array([[r["recording_level_split_percent"], r["participant_disjoint_split_percent"]] for r in rows])
    if not np.allclose(actual, expected, atol=0.005, rtol=0):
        raise AssertionError(f"LSTM reports differ from the manuscript:\n{actual}")
    files = {"tab:rnn-split": [write_table(output, "tab-rnn-split", rows)]}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, filename, title in zip(
        axes, ("accuracy_curve_11.png", "loss_curve_11.png"), ("(a) Accuracy", "(b) Loss")
    ):
        ax.imshow(plt.imread(LSTM_ASSETS / filename))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    files["fig:rnn-leakage-curves"] = []
    for suffix in ("pdf", "png"):
        target = output / f"fig-rnn-leakage-curves.{suffix}"
        fig.savefig(target, dpi=220)
        files["fig:rnn-leakage-curves"].append(target.name)
    plt.close(fig)
    return files


def export_static_audits(output: Path, cache: Path) -> dict[str, list[str]]:
    files = {}
    files["tab:audit-replay"] = [write_table(output, "tab-audit-replay", replay_audit(cache))]
    expanded = [
        ("nguyen2022", "Recording paths", "Shuffled per-file boundaries followed by ungrouped fold slices", "Published folds not reconstructed"),
        ("naimi2023hct", "Recording paths", "Shuffled per-file boundaries followed by ungrouped fold slices", "Published folds not reconstructed"),
        ("naimi2025inceptoformer", "Recording paths", "Shuffled per-file boundaries followed by ungrouped fold slices", "Published folds not reconstructed"),
        ("zhang2025benchmark", "Recording paths", "Adapted Nguyen per-file boundary and fold path", "Shared implementation lineage"),
        ("li2026subgaitnet", "Recording paths", "Shuffled KFold over the recording-file list", "Protocol-code consistency finding"),
        ("vimalajeewa2023", "Feature rows", "Class-wise randperm without groups", "Row-to-participant map unavailable"),
        ("nguyen2025motion", "Gait segments", "random_split and ordinary StratifiedKFold", "Training recordings not released"),
        ("zhao2022astcapsnet", "Array rows", "train_test_split without groups", "Precomputed gait arrays unavailable"),
        ("mittra2018", "Transformed rows", "train_test_split without groups", "Row-to-participant provenance incomplete"),
        ("zhao2022corrmnn", "Array rows", "Paired train_test_split calls without groups", "Local precomputed arrays unavailable"),
    ]
    files["tab:expanded-audit"] = [write_table(output, "tab-expanded-audit", [
        {"study": s, "split_unit": unit, "released_operation": operation, "reconstruction_limit": limit}
        for s, unit, operation, limit in expanded
    ])]
    provenance = [
        ("Data loading", "generateLPresidual.m lines 4-31; Original.ipynb cells 2-4 and 6", "load_inputs lines 519-547"),
        ("Global LP fitting", "generateLPresidual.m lines 42-59 and 63-95", "fit_lp/apply_lp lines 550-568; factorial lines 823-825"),
        ("Fold-local LP fitting", "No fold-local path in the released preprocessing script", "features lines 640-652; factorial lines 827-828"),
        ("Outer folds and outer-test training control", "Original.ipynb cell 8, lines 1, 16-24, 87-89, and 132-134", "outer_folds lines 581-593; subject_split lines 571-578; factorial line 822"),
        ("Early stopping and LR scheduling", "Original.ipynb cell 8, lines 74-89 and 120-134", "safe_fit lines 439-516"),
        ("Two-stage training", "Original.ipynb cell 8, lines 52-71, 73-94, and 98-134", "build_lpgnet lines 114-128; build_recording_model lines 148-153; train lines 660-690"),
        ("Evaluation and aggregation", "Original.ipynb cell 8, lines 139-158 and 166-168", "predict lines 692-718; notebook cell 6; summary_tables lines 921-930"),
    ]
    files["tab:lpgnet-provenance"] = [write_table(output, "tab-lpgnet-provenance", [
        {"operation": op, "released_implementation": released, "reconstruction_implementation": reconstruction}
        for op, released, reconstruction in provenance
    ])]
    return files


def export_cohort_map(output: Path, direct: dict, factorial: dict) -> list[str]:
    included = [r for r in direct["audit"] if r["status"] == "included"]
    direct_participants = [p for r in direct["runs"] if r["family"] == "cnn" and r["seed"] == 42 for p in r["test_participants"]]
    factorial_participants = factorial["participants"]
    factorial_participants = factorial_participants[factorial_participants["condition"] == "global__test"]
    rows = [
        {"analysis": "Demographic description", "recordings": "", "participants": 166,
         "pd": 93, "control": 73, "protocol_role": "Metadata-level cohort description"},
        {"analysis": "LPGNet audit and factorial", "recordings": 306,
         "participants": len(factorial_participants), "pd": int(factorial_participants["label"].sum()),
         "control": int((1 - factorial_participants["label"]).sum()),
         "protocol_role": "Official sensor inventory; fold seed 0; one factorial seed schedule"},
        {"analysis": "Direct CNN and Mamba", "recordings": len(included),
         "participants": len(direct_participants), "pd": sum(p["label"] for p in direct_participants),
         "control": sum(1 - p["label"] for p in direct_participants),
         "protocol_role": "Curated inventory; split seed 42; three training seeds"},
    ]
    expected = [(306, 165, 93, 72), (292, 161, 90, 71)]
    actual = [(int(r["recordings"]), r["participants"], r["pd"], r["control"]) for r in rows[1:]]
    if actual != expected:
        raise AssertionError(f"Cohort map differs: {actual}")
    return [write_table(output, "tab-cohort-map", rows)]


def export_direct(output: Path, direct: dict, clinical: dict) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    seeds = direct["protocol"]["training_seeds"]
    families = ("cnn", "mamba")

    rows = []
    for family in families:
        seed_rows = [direct["summaries"][family][str(seed)]["participant_metrics"] for seed in seeds]
        row = {"model": family.upper()}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity"):
            row[f"{metric}_percent"] = 100 * np.mean([r[metric] for r in seed_rows])
        for metric in ("roc_auc", "log_loss"):
            row[metric] = np.mean([r[metric] for r in seed_rows])
        rows.append(row)
    files["tab:direct-results"] = [write_table(output, "tab-direct-results", rows)]

    participant_rows, window_rows = [], []
    for family in families:
        for seed in seeds:
            summary = direct["summaries"][family][str(seed)]
            for target, level in ((participant_rows, "participant_metrics"), (window_rows, "window_metrics")):
                metrics = summary[level]
                row = {"model": family.upper(), "seed": seed}
                if level == "window_metrics":
                    row["windows"] = metrics["n"]
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity"):
                    row[f"{metric}_percent"] = 100 * metrics[metric]
                row["roc_auc"] = metrics["roc_auc"]
                row["log_loss"] = metrics["log_loss"]
                if level == "participant_metrics":
                    row.update({key: metrics[key] for key in ("tn", "fp", "fn", "tp")})
                target.append(row)
    files["tab:direct-participant"] = [write_table(output, "tab-direct-participant", participant_rows)]
    files["tab:direct-window"] = [write_table(output, "tab-direct-window", window_rows)]

    for family in families:
        family_runs = [r for r in direct["runs"] if r["family"] == family]
        selected = Counter(r["trials"][r["selected_candidate"]]["candidate"]["name"] for r in family_runs)
        first_trials = {t["candidate"]["name"]: t for t in family_runs[0]["trials"]}
        rows = []
        for candidate in direct["candidates"][family]:
            trial = first_trials[candidate["name"]]
            rows.append({
                "id": candidate["name"], "width": candidate["width"], "layers": candidate["layers"],
                "kernel": candidate["kernel"], "state": candidate["state"], "context": candidate["seq_len"],
                "normalization": candidate["normalization"], "pooling": candidate["pooling"],
                "dropout": candidate["dropout"], "learning_rate": candidate["lr"],
                "weight_decay": candidate["weight_decay"], "parameters": trial["parameters"],
                "receptive_field_samples": trial["receptive_field_samples"],
                "selected": selected[candidate["name"]],
                "median_selected_epoch_all_candidates": np.median([
                    r["trials"][r["selected_candidate"]]["best_epoch"] for r in family_runs
                ]),
            })
        if sum(selected.values()) != 30:
            raise AssertionError("Selected-candidate counts must cover 30 tasks")
        files[f"tab:{family}-candidates"] = [write_table(output, f"tab-{family}-candidates", rows)]

    reports = diagnostics.build_diagnostics(direct, clinical)
    severity = pd.DataFrame(reports["pd_severity"])
    rows = []
    for stage in ("2.0", "2.5", "3.0"):
        stage_rows = severity[severity["hy_group"] == stage]
        row = {"hy_stage": stage, "pd_participants": int(stage_rows["n"].max())}
        for family in families:
            group = stage_rows[stage_rows["family"] == family]
            if sorted(group["seed"].tolist()) != sorted(seeds) or group["n"].nunique() != 1:
                raise AssertionError("Incomplete HY seed coverage")
            values = 100 * group["sensitivity"].to_numpy(dtype=float)
            row[f"{family}_sensitivity_percent"] = values.mean()
            row[f"{family}_sensitivity_min_percent"] = values.min()
            row[f"{family}_sensitivity_max_percent"] = values.max()
            row[f"{family}_mean_pd_probability"] = group["mean_pd_probability"].mean()
        rows.append(row)
    files["tab:direct-hy"] = [write_table(output, "tab-direct-hy", rows)]

    study = pd.DataFrame(reports["by_study_diagnosis"])
    rows = []
    for (study_name, diagnosis_name), group in study.groupby(["study", "diagnosis"]):
        row = {"study": study_name, "diagnosis": diagnosis_name, "participants": int(group["n"].max())}
        for family in families:
            family_rows = group[group["family"] == family]
            if sorted(family_rows["seed"].tolist()) != sorted(seeds) or family_rows["n"].nunique() != 1:
                raise AssertionError("Incomplete study/seed coverage")
            row[f"{family}_correct_percent"] = 100 * family_rows["accuracy"].mean()
        rows.append(row)
    files["tab:direct-study"] = [write_table(output, "tab-direct-study", rows)]

    by_fold = pd.DataFrame(reports["by_fold"])
    rows = []
    for split in direct["splits"]:
        fold = int(split["fold"])
        test = split["test"]
        pd_subjects = [s for s in test if clinical[s]["metadata_label"] == 1]
        advanced = sum(clinical[s]["hy"] in (2.5, 3.0) for s in pd_subjects)
        row = {"fold": fold, "participants": len(test),
               "advanced_pd": advanced, "pd_participants": len(pd_subjects)}
        for family in families:
            group = by_fold[(by_fold["family"] == family) & (by_fold["fold"] == fold)]
            if sorted(group["seed"].tolist()) != sorted(seeds):
                raise AssertionError("Incomplete fold/seed coverage")
            row[f"{family}_accuracy_percent"] = 100 * group["accuracy"].mean()
        rows.append(row)
    files["tab:direct-folds"] = [write_table(output, "tab-direct-folds", rows)]

    stability = pd.DataFrame(reports["seed_stability"])
    always = {family: set(stability[(stability["family"] == family) & stability["always_wrong"]]["subject"])
              for family in families}
    persistent = sorted(always["cnn"] & always["mamba"])
    rows = []
    for subject in persistent:
        row = stability[(stability["family"] == "cnn") & (stability["subject"] == subject)].iloc[0]
        rows.append({"subject": subject, "diagnosis": "PD" if row["label"] else "Control",
                     "hy": row["hy"] if row["label"] else ""})
    if (sum(r["diagnosis"] == "Control" for r in rows), sum(r["diagnosis"] == "PD" for r in rows)) != (10, 10):
        raise AssertionError("Persistent-error inventory differs")
    files["tab:persistent-errors"] = [write_table(output, "tab-persistent-errors", rows)]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_paper")
    parser.add_argument("--cache", type=Path, default=Path(".cache/physionet"),
                        help="Checksum-verified PhysioNet manifest/metadata cache")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    direct = read_direct()
    factorial = read_factorial()
    clinical, _ = load_verified_clinical(args.cache, download=True)
    artifacts: dict[str, list[str]] = {}
    artifacts["tab:cohort-map"] = export_cohort_map(args.output, direct, factorial)
    artifacts.update(export_static_audits(args.output, args.cache))
    artifacts.update(export_lstm(args.output))
    artifacts.update(export_lpgnet(args.output, factorial))
    artifacts["fig:lpgnet-loss"] = export_loss_figure(args.output)
    artifacts.update(export_direct(args.output, direct, clinical))

    expected = {
        "tab:cohort-map", "tab:audit-replay", "tab:expanded-audit", "tab:rnn-split",
        "fig:rnn-leakage-curves", "tab:lpgnet-replication", "tab:factorial-protocols",
        "tab:lpgnet-factorial", "tab:factorial-contrasts", "fig:lpgnet-roc",
        "fig:lpgnet-confusion", "tab:direct-results", "tab:direct-hy", "tab:direct-study",
        "tab:lpgnet-provenance", "tab:cnn-candidates", "tab:mamba-candidates",
        "tab:direct-participant", "tab:direct-window", "tab:direct-folds",
        "tab:persistent-errors", "fig:lpgnet-loss",
    }
    if set(artifacts) != expected:
        raise AssertionError(f"Artifact map mismatch: missing={expected-set(artifacts)}, extra={set(artifacts)-expected}")
    manifest = {
        "schema": 1,
        "bootstrap": {"samples": N_BOOT, "seed": BOOTSTRAP_SEED},
        "inputs": {
            "direct_archive_sha256": hashlib.sha256(DIRECT_ARCHIVE.read_bytes()).hexdigest(),
            "factorial_archive_sha256": hashlib.sha256(FACTORIAL_ARCHIVE.read_bytes()).hexdigest(),
            "factorial_histories_sha256": hashlib.sha256((LPGNET_ASSETS / "recording_loss_histories.json").read_bytes()).hexdigest(),
        },
        "artifacts": artifacts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(artifacts)} manuscript labels to {args.output.resolve()}")
    for label in sorted(artifacts):
        print(f"{label}: {', '.join(artifacts[label])}")


if __name__ == "__main__":
    main()
