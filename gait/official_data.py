"""Opt-in verified PhysioNet inputs and conservative cross-identity quarantine."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import io
from pathlib import Path
import re
import urllib.request

import numpy as np

BASE = "https://physionet.org/files/gaitpdb/1.0.0/"
MANIFEST_SHA256 = "44903ce52d5428ba218a9cdfd60d5646ab6674df4926233c756ae7c3c7cd50ec"
RECORDING = re.compile(r"^(Ga|Ju|Si)(Co|Pt)\d{2}_\d{2}\.txt$")
PARTICIPANT = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d{2})$")


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def verified_file(cache, name, expected, download=True):
    """Use an exact checksum-matching cache; never trust a similarly named input."""
    if Path(name).name != name:
        raise ValueError("Only manifest basenames are accepted")
    path = Path(cache) / name
    blob = path.read_bytes() if path.is_file() else b""
    if sha256(blob) != expected:
        if not download:
            raise ValueError(f"Missing or checksum-mismatched cached file: {name}")
        with urllib.request.urlopen(BASE + name, timeout=120) as response:
            blob = response.read()
        if sha256(blob) != expected:
            raise ValueError(f"Official checksum mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return blob


def manifest(cache, download=True):
    blob = verified_file(cache, "SHA256SUMS.txt", MANIFEST_SHA256, download)
    entries = {}
    for line in blob.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if name in entries:
            raise ValueError("Repeated manifest name")
        entries[name] = digest
    if sum(bool(RECORDING.fullmatch(n)) for n in entries) != 306:
        raise ValueError("Unexpected official recording inventory")
    return entries


def curate_inventory(inventory, signals, min_samples):
    """Identify all conflicts BEFORE short exclusions or within-ID deduplication."""
    groups = defaultdict(list)
    for row in inventory:
        groups[row["force_sha256"]].append(row)
    conflicts = [g for g in groups.values() if len({r["subject"] for r in g}) > 1]
    excluded = {r["subject"] for group in conflicts for r in group}
    seen, records, audit = {}, [], []
    for original in sorted(inventory, key=lambda r: r["path"]):
        row = dict(original)
        if row["subject"] in excluded:
            row["status"] = "participant_excluded_identity_ambiguity"
        elif row["force_sha256"] in seen:
            row.update(status="duplicate_excluded", duplicate_of=seen[row["force_sha256"]])
        else:
            seen[row["force_sha256"]] = row["path"]
            row["status"] = "short_excluded" if row["samples"] < min_samples else "included"
            if row["status"] == "included":
                signal = signals[row["path"]]
                if not np.isfinite(signal).all():
                    raise ValueError("Invalid float32 force values")
                records.append({**row, "signal": signal})
        audit.append(row)
    provenance = dict(source=BASE, manifest_sha256=MANIFEST_SHA256,
                      min_samples=min_samples, excluded_subjects=sorted(excluded),
                      conflict_groups=[[r["path"] for r in g] for g in conflicts],
                      policy="exclude_all_recordings_of_cross_identity_duplicate_participants")
    return records, audit, provenance


def load_verified_cohort(cache, min_samples=2000, download=True, workers=4):
    entries = manifest(cache, download)

    def load(name):
        blob = verified_file(cache, name, entries[name], download)
        values = np.loadtxt(io.BytesIO(blob), dtype=np.float64, ndmin=2)
        if values.shape[1] != 19 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid original matrix: {name}")
        dt = np.diff(values[:, 0])
        if not np.all(dt > 0) or not np.allclose(dt, .01, atol=.002, rtol=0):
            raise ValueError(f"Invalid 100-Hz time axis: {name}")
        forces = np.ascontiguousarray(values[:, 1:], dtype="<f8")
        row = dict(path=name, recording=name[:-4], subject=name[:6], study=name[:2],
                   label=int(name[2:4] == "Pt"), samples=len(values),
                   sha256=entries[name], force_sha256=sha256(forces.tobytes()))
        return row, forces.astype(np.float32)

    names = sorted(n for n in entries if RECORDING.fullmatch(n))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(pool.map(load, names))
    return curate_inventory([r for r, _ in pairs], {r["path"]: x for r, x in pairs}, min_samples)


class ClinicalHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def clinical_number(raw):
    if raw.strip().lower() in ("", "nan", "-", "–", "—", "na"):
        return None
    value = float(raw)
    return value if np.isfinite(value) else None


def parse_clinical(text_blob, html_blob):
    """Exact-ID join; independently cross-check HY in two official formats.

    Only the first nine TXT columns are used: some later fields have missing
    delimiters. Never repair IDs from numeric substrings or infer HY from names.
    """
    parser = ClinicalHTML()
    parser.feed(html_blob.decode("utf-8"))
    if not parser.rows or parser.rows[0][:3] != ["ID", "Study", "Group"] or parser.rows[0][8] != "HoehnYahr":
        raise ValueError("Unexpected official HTML metadata schema")
    html_rows = defaultdict(list)
    for row in parser.rows[1:]:
        if len(row) >= 9:
            html_rows[row[0]].append(row)
    lines = [l.split() for l in text_blob.decode("utf-8-sig").splitlines() if l.strip()]
    if lines[0][:9] != ["ID", "Study", "Group", "Subjnum", "Gender", "Age", "Height", "Weight", "HoehnYahr"]:
        raise ValueError("Unexpected official TXT metadata schema")
    indexed = defaultdict(list)
    issues = []
    for row in lines[1:]:
        if len(row) < 9 or not PARTICIPANT.fullmatch(row[0]):
            issues.append(dict(raw_id=row[0], reason="invalid_exact_id_or_short_row"))
            continue
        indexed[row[0]].append(row)
    clinical = {}
    for subject, rows in indexed.items():
        row = rows[0]
        match = PARTICIPANT.fullmatch(subject)
        label = 1 if row[2] == "1" else 0 if row[2] == "2" else None
        status = "verified"
        if len(rows) != 1 or row[1] != match[1] or label != int(match[2] == "Pt") or int(row[3]) != int(match[3]):
            status = "identity_or_label_conflict"
        html = html_rows.get(subject, [])
        hy = clinical_number(row[8])
        if len(html) != 1:
            status = "html_id_missing_or_ambiguous" if status == "verified" else status
        else:
            h = html[0]
            html_label = 1 if h[2].upper() == "PD" else 0 if h[2].upper() in ("CO", "HC", "CONTROL") else None
            if h[1] != row[1] or html_label != label or int(h[3]) != int(row[3]) or clinical_number(h[8]) != hy:
                status = "clinical_formats_disagree" if status == "verified" else status
        if hy is not None and hy not in (0, 1, 1.5, 2, 2.5, 3, 4, 5):
            status = "invalid_hy_value"
        if status != "verified":
            issues.append(dict(raw_id=subject, reason=status))
        clinical[subject] = dict(subject=subject, metadata_label=label, metadata_study=row[1],
                                 hy=hy if status == "verified" else None,
                                 hy_status=status if status != "verified" else "missing" if hy is None else "verified",
                                 age=clinical_number(row[5]) if status == "verified" else None,
                                 gender_code=row[4] if status == "verified" else None)
    return clinical, issues


def load_verified_clinical(cache, download=True):
    entries = manifest(cache, download)
    names = ("demographics.txt", "demographics.html")
    blobs = [verified_file(cache, n, entries[n], download) for n in names]
    clinical, issues = parse_clinical(*blobs)
    provenance = dict(source=BASE, manifest_sha256=MANIFEST_SHA256,
                      files={n: entries[n] for n in names}, issues=issues,
                      join="exact study-qualified ID; TXT/HTML HY and identity cross-check")
    return clinical, provenance