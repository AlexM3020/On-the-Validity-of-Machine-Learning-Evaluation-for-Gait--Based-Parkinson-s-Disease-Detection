"""Training, checkpoint evaluation, and partition provenance for gait models."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .data import GaitDataset, discover_recordings, split_recordings
from .models import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dataset_manifest(dataset, root):
    """Record relative filenames, hashes, participants, labels, and windows."""
    return {
        "subjects": sorted({record.subject for record in dataset.recordings}),
        "recordings": [
            {"file": record.path.relative_to(root).as_posix(), "subject": record.subject,
             "label": record.label, "sha256": hashlib.sha256(record.path.read_bytes()).hexdigest()}
            for record in dataset.recordings
        ],
        "windows": [
            {"file": window.record.path.relative_to(root).as_posix(),
             "start": window.start, "length": window.valid_length}
            for window in dataset.windows
        ],
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_weight = 0.0, 0.0
    targets, predictions = [], []
    probabilities = []
    with torch.set_grad_enabled(optimizer is not None):
        for values, labels, lengths in loader:
            values, labels = values.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(values, lengths)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss; check data and configuration")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            weight = (criterion.weight[labels].sum().item()
                      if criterion.weight is not None else len(labels))
            total_loss += loss.item() * weight
            total_weight += weight
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            probabilities.extend(logits.softmax(1)[:, 1].detach().cpu().tolist())
    if total_weight == 0:
        raise ValueError("Cannot evaluate an empty partition")
    # Exact weighted-CE mean, independent of evaluation batch boundaries.
    return total_loss / total_weight, targets, predictions, probabilities


def report(targets, predictions):
    result = classification_report(targets, predictions, labels=[0, 1],
                                   target_names=["Control", "Parkinson"],
                                   output_dict=True, zero_division=0)
    result["confusion_matrix"] = confusion_matrix(targets, predictions, labels=[0, 1]).tolist()
    result["evaluation_unit"] = "window"
    return result


def save_evaluation(output, name, dataset, root, outcome):
    loss, targets, predictions, probabilities = outcome
    result = report(targets, predictions)
    result["loss"] = loss
    write_json(output / f"{name}_report.json", result)
    write_json(output / f"{name}_predictions.json", [
        {"file": window.record.path.relative_to(root).as_posix(),
         "subject": window.record.subject, "start": window.start,
         "target": target, "prediction": prediction, "pd_probability": probability}
        for window, target, prediction, probability in
        zip(dataset.windows, targets, predictions, probabilities)
    ])


def plot_history(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for partition in ("train", "validation"):
        ax.plot([row["epoch"] for row in history],
                [row[f"{partition}_loss"] for row in history], label=partition)
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160)
    plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description="Subject-independent gait classification")
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--model", choices=["mamba", "lstm", "gru"], default="mamba")
    result.add_argument("--seq-len", type=int, default=2000)
    result.add_argument("--channels", type=int, nargs="+", default=list(range(1, 19)),
                        help="Zero-based sensor columns; default 1..18 excludes time")
    result.add_argument("--hidden-dim", type=int, default=128)
    result.add_argument("--num-layers", type=int, default=2)
    result.add_argument("--d-state", type=int, default=32)
    result.add_argument("--d-conv", type=int, default=32)
    result.add_argument("--expand", type=int, default=2)
    result.add_argument("--dropout", type=float, default=0.0)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--lr", type=float, default=5e-5)
    result.add_argument("--class-weights", type=float, nargs=2, default=[1.1, 1.0],
                        metavar=("CONTROL", "PD"))
    result.add_argument("--patience", type=int, default=10)
    result.add_argument("--scheduler-patience", type=int, default=5)
    result.add_argument("--validation-fraction", type=float, default=0.2)
    result.add_argument("--test-fraction", type=float, default=0.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    result.add_argument("--checkpoint", type=Path,
                        help="Evaluate a checkpoint created by this release; no training")
    result.add_argument("--partition", choices=["validation", "test"], default="validation")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    for name in ("batch_size", "epochs", "patience"):
        if getattr(args, name) <= 0:
            argument_parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.scheduler_patience < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        argument_parser.error("Learning rate must be positive and scheduler patience nonnegative")
    if any(not math.isfinite(weight) or weight <= 0 for weight in args.class_weights):
        argument_parser.error("Class weights must be finite and positive")
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        argument_parser.error("Output directory must be empty; use a new directory for each run")
    root = args.data_dir.resolve()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        argument_parser.error("CUDA requested but unavailable")
    checkpoint = None
    if args.checkpoint:
        # Restricted tensor/state loading, not arbitrary full-model unpickling.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        config = checkpoint["configuration"]
    else:
        config = {key: value for key, value in vars(args).items()
                  if key not in {"data_dir", "output_dir", "checkpoint", "partition", "device"}}
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config["model"] == "mamba" and device != "cuda":
        argument_parser.error("Mamba requires CUDA on Linux/WSL; use --model lstm or gru on CPU")
    records = discover_recordings(root)
    partitions = split_recordings(records, config["validation_fraction"], config["test_fraction"], config["seed"])
    datasets = {name: GaitDataset(group, config["seq_len"], config["channels"])
                for name, group in partitions.items()}
    manifest = {name: dataset_manifest(dataset, root) for name, dataset in datasets.items()}
    if checkpoint is not None and manifest != checkpoint["split_manifest"]:
        raise ValueError("Dataset content or split differs from the checkpoint manifest")
    if checkpoint is not None and args.partition not in datasets:
        argument_parser.error("Checkpoint has no requested partition")
    model = build_model(config["model"], len(config["channels"]), config["hidden_dim"],
                        config["num_layers"], config["d_state"], config["d_conv"],
                        config["expand"], config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config["class_weights"], device=device))
    loaders = {name: DataLoader(dataset, batch_size=config["batch_size"],
                                shuffle=name == "train" and checkpoint is None,
                                generator=torch.Generator().manual_seed(config["seed"]))
               for name, dataset in datasets.items()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration.json", dict(config, device=device,
               torch_version=str(torch.__version__), numpy_version=str(np.__version__)))
    write_json(output / "split_manifest.json", manifest)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        outcome = run_epoch(model, loaders[args.partition], criterion, device)
        save_evaluation(output, args.partition, datasets[args.partition], root, outcome)
        print(json.dumps(report(outcome[1], outcome[2]), indent=2))
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config["scheduler_patience"])
    best_loss, stale = float("inf"), 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        train = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation = run_epoch(model, loaders["validation"], criterion, device)
        row = {"epoch": epoch, "train_loss": train[0], "validation_loss": validation[0],
               "train_accuracy": report(train[1], train[2])["accuracy"],
               "validation_accuracy": report(validation[1], validation[2])["accuracy"],
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        scheduler.step(validation[0])
        if validation[0] < best_loss:
            best_loss, stale = validation[0], 0
            torch.save({"format_version": 1, "model_state": model.state_dict(),
                        "configuration": config, "split_manifest": manifest,
                        "epoch": epoch}, output / "best_model.pt")
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    best = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    for name in ("validation", "test"):
        if name in datasets:
            save_evaluation(output, name, datasets[name], root,
                            run_epoch(model, loaders[name], criterion, device))
    plot_history(output, history)


if __name__ == "__main__":
    main()