"""Post-evaluation diagnostics. Clinical metadata never enters model selection."""

from collections import defaultdict
import math
import numpy as np
from . import cross_validation as cv


def participant_errors(result, clinical, partition="test"):
    if not result.get("complete"):
        raise ValueError("Complete both experiments before examining outer errors")
    if partition not in ("test", "validation"):
        raise ValueError("Unknown partition")
    errors = []
    for run in result["runs"]:
        selected = run["trials"][run["selected_candidate"]]
        participants = run["test_participants"] if partition == "test" else selected["validation_participants"]
        for p in participants:
            subject = p["subject"]
            meta = clinical.get(subject)
            status = "unmatched" if meta is None else meta["hy_status"]
            hy = None if meta is None else meta["hy"]
            if meta is not None and (meta["metadata_label"] != p["label"] or meta["metadata_study"] != subject[:2]):
                status, hy = "prediction_metadata_identity_conflict", None
            probability = p["probability"]
            prediction = int(probability >= .5)
            true_probability = probability if p["label"] else 1-probability
            correct = prediction == p["label"]
            confidence = max(probability, 1-probability)
            errors.append(dict(family=run["family"], seed=run["seed"], fold=run["fold"],
                               partition=partition, subject=subject, study=subject[:2],
                               label=p["label"], diagnosis="PD" if p["label"] else "Control",
                               probability=probability, prediction=prediction, correct=correct,
                               recordings=p["recordings"], windows=p["windows"], hy=hy,
                               hy_status=status, age=meta.get("age") if meta else None,
                               confidence=confidence, confident_error=not correct and confidence >= .8,
                               nll=-math.log(max(true_probability, 1e-15)),
                               candidate=selected["candidate"]["name"]))
    return errors


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0, center-half), min(1, center+half)


def subgroup_table(rows, keys):
    """Separate model/seed groups; singleton-class accuracy is sensitivity/specificity."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["seed"], *(r.get(k) for k in keys))].append(r)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        n = len(group)
        if len({r["subject"] for r in group}) != n:
            raise ValueError("Repeated participants in subgroup; do not pool validation folds or seeds")
        correct = sum(r["correct"] for r in group)
        low, high = wilson(correct, n)
        labels = {r["label"] for r in group}
        row = dict(family=key[0], seed=key[1], **dict(zip(keys, key[2:])),
                   n=n, correct=correct, errors=n-correct, accuracy=correct/n,
                   accuracy_wilson_lower=low, accuracy_wilson_upper=high,
                   mean_nll=float(np.mean([r["nll"] for r in group])),
                   mean_pd_probability=float(np.mean([r["probability"] for r in group])),
                   sensitivity=correct/n if labels == {1} else None,
                   specificity=correct/n if labels == {0} else None,
                   roc_auc=None, balanced_accuracy=None)
        if labels == {0, 1}:
            row.update({k: v for k, v in cv.metrics(group).items() if k in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity")})
        output.append(row)
    return output


def seed_stability(rows, expected_seeds):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["subject"])].append(r)
    output = []
    for (family, subject), group in sorted(groups.items()):
        if len(group) != len(expected_seeds) or {r["seed"] for r in group} != set(expected_seeds):
            raise ValueError("Seed coverage mismatch")
        if len({r["fold"] for r in group}) != 1 or len({r["label"] for r in group}) != 1:
            raise ValueError("Fold/label changed across training seeds")
        probabilities = [r["probability"] for r in group]
        correct = sum(r["correct"] for r in group)
        output.append(dict(family=family, subject=subject, study=group[0]["study"], label=group[0]["label"],
                           hy=group[0]["hy"], hy_status=group[0]["hy_status"], fold=group[0]["fold"],
                           n_seeds=len(group), correct_seeds=correct, error_seeds=len(group)-correct,
                           always_wrong=correct == 0, predictions_disagree=len({r["prediction"] for r in group}) > 1,
                           probability_mean=float(np.mean(probabilities)),
                           probability_std=float(np.std(probabilities, ddof=1)) if len(group) > 1 else None,
                           probability_min=min(probabilities), probability_max=max(probabilities)))
    return output


def paired_comparison(rows, bootstrap_samples=2000, seed=8107):
    """Paired stratified participant bootstrap; also paired across repeated seeds.

    Reports Mamba minus CNN for the average of seed-specific metric differences,
    not an ensemble. Does not treat repeated seed rows as independent people.
    """
    seeds = sorted({r["seed"] for r in rows})
    by_key = {(r["family"], r["seed"], r["subject"]): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate OOF keys")
    subjects = sorted({r["subject"] for r in rows})
    a, b = [], []
    for s in seeds:
        a.append([by_key[("mamba", s, p)] for p in subjects])
        b.append([by_key[("cnn", s, p)] for p in subjects])
    for ar, br in zip(a, b):
        for first, second in zip(ar, br):
            if first["label"] != second["label"] or first["fold"] != second["fold"]:
                raise ValueError("Models do not share labels/folds")
    labels = np.array([r["label"] for r in a[0]])
    groups = [np.flatnonzero(labels == i) for i in (0, 1)]
    names = ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc")

    def differences(idx):
        delta = []
        for ar, br in zip(a, b):
            am, bm = cv.metrics([ar[i] for i in idx]), cv.metrics([br[i] for i in idx])
            delta.append([am[k]-bm[k] for k in names])
        return np.mean(delta, axis=0)

    point = differences(np.arange(len(subjects)))
    rng = np.random.default_rng(seed)
    samples = np.array([differences(np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]))
                        for _ in range(bootstrap_samples)])
    return [dict(metric=name, mamba_minus_cnn=float(point[i]),
                 lower=float(np.percentile(samples[:, i], 2.5)), upper=float(np.percentile(samples[:, i], 97.5)),
                 n_participants=len(subjects), n_seeds=len(seeds)) for i, name in enumerate(names)]


def build_diagnostics(result, clinical):
    outer = participant_errors(result, clinical)
    inner = participant_errors(result, clinical, "validation")
    pd = [{**r, "hy_group": str(r["hy"]) if r["hy_status"] == "verified" and r["hy"] is not None else "Unknown"}
          for r in outer if r["label"] == 1]
    # All expected stages appear even when absent, so a missing subgroup is explicit.
    severity = subgroup_table(pd, ["hy_group"])
    for family in result["candidates"]:
        for seed in result["protocol"]["training_seeds"]:
            for stage in ("2.0", "2.5", "3.0", "Unknown"):
                if not any((r["family"], r["seed"], r["hy_group"]) == (family, seed, stage) for r in severity):
                    severity.append(dict(family=family, seed=seed, hy_group=stage, n=0, correct=0,
                                         errors=0, sensitivity=None, accuracy_wilson_lower=None, accuracy_wilson_upper=None))
    return dict(participant_errors=outer, inner_validation_errors=inner,
                by_study=subgroup_table(outer, ["study"]),
                by_study_diagnosis=subgroup_table(outer, ["study", "diagnosis"]),
                by_fold=subgroup_table(outer, ["fold"]), pd_severity=severity,
                pd_severity_study=subgroup_table(pd, ["study", "hy_group"]),
                metadata_coverage=subgroup_table(outer, ["diagnosis", "hy_status"]),
                confidently_wrong=sorted([r for r in outer if r["confident_error"]], key=lambda r: -r["nll"]),
                inner_high_loss=sorted(inner, key=lambda r: -r["nll"]),
                seed_stability=seed_stability(outer, result["protocol"]["training_seeds"]),
                paired_comparison=paired_comparison(outer, result["protocol"]["bootstrap_samples"],
                                                     result["protocol"]["bootstrap_seed"]))