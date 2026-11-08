"""Metadata-only control-branch audit; no sensor loading, fitting, or upstream code."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from urllib.request import urlopen


MANIFEST_URL = 'https://physionet.org/files/gaitpdb/1.0.0/SHA256SUMS.txt'
MANIFEST_SHA256 = '44903ce52d5428ba218a9cdfd60d5646ab6674df4926233c756ae7c3c7cd50ec'
DEFAULT_MANIFEST = Path(__file__).parent / 'fixtures' / 'SHA256SUMS.txt'
MAX_MANIFEST_BYTES = 1_000_000
RECORD = re.compile(
    r'(?P<subject>(?:Ga|Ju|Si)(?:Co|Pt)[0-9]{2})_'
    r'(?P<record>[0-9]{2})(?:_[A-Za-z0-9-]+)*\.txt'
)
CHECKSUM_LINE = re.compile(r'([0-9a-fA-F]{64})[ \t]+\*?([^\r\n]+)')


def recording_identity(filename):
    """Return full study-qualified participant ID and parent recording filename.

    Both path separators and suffixes such as _chunk_001 or _segment_2 are
    accepted. Chunk names must retain the original subject_record prefix.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError('recording filename must be a nonempty string')
    name = filename.replace('\\', '/').rsplit('/', 1)[-1]
    match = RECORD.fullmatch(name)
    if match is None:
        raise ValueError('expected study(Co|Pt)NN_RR[_chunk_suffix].txt; study is Ga, Ju, or Si')
    subject = match['subject']
    return subject, f"{subject}_{match['record']}.txt"


def subject_id(filename):
    return recording_identity(filename)[0]


def parse_manifest(content):
    """Parse checksum rows, retaining sensor names/hashes and rejecting duplicates."""
    records = []
    seen = set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        match = CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f'invalid SHA256SUMS row {line_number}')
        checksum, name = match.groups()
        if name in seen:
            raise ValueError(f'duplicate manifest filename at row {line_number}')
        seen.add(name)
        # Official auxiliary files are not sensor recordings.
        if RECORD.fullmatch(name) is None:
            continue
        participant, recording = recording_identity(name)
        records.append({'filename': name, 'subject_id': participant,
                        'recording_filename': recording, 'sha256': checksum.lower()})
    if not records:
        raise ValueError('manifest contains no matching sensor recordings')
    return sorted(records, key=lambda row: row['filename'])


def load_manifest(path=DEFAULT_MANIFEST, download=False):
    """Network is opt-in and memory-only; never download raw recordings."""
    if download:
        with urlopen(MANIFEST_URL, timeout=30) as response:
            payload = response.read(MAX_MANIFEST_BYTES + 1)
        source = MANIFEST_URL
    else:
        with Path(path).open('rb') as handle:
            payload = handle.read(MAX_MANIFEST_BYTES + 1)
        source = 'local_manifest'
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError('manifest exceeds size limit')
    digest = hashlib.sha256(payload).hexdigest()
    pinned = digest == MANIFEST_SHA256
    if not download and Path(path).resolve() == DEFAULT_MANIFEST.resolve() and not pinned:
        raise ValueError('bundled manifest checksum mismatch; restore the official fixture')
    return parse_manifest(payload.decode('utf-8')), {
        'source': source, 'sha256': digest, 'bytes': len(payload),
        'matches_bundled_official_sha256': pinned,
        'raw_recordings_downloaded': False,
    }


def _validate_controls(files, fold, folds):
    if not isinstance(folds, int) or folds < 2:
        raise ValueError('fold count must be an integer >= 2')
    if not isinstance(fold, int) or not 0 <= fold < folds:
        raise ValueError('fold must be in [0, fold count)')
    if not files or len(files) != len(set(files)):
        raise ValueError('control filenames must be nonempty and unique')
    if any(subject_id(name)[2:4] != 'Co' for name in files):
        raise ValueError('this replay accepts the CONTROL BRANCH ONLY')


def replay_control_fold(files, seed=42, fold=0, folds=10):
    """Replay recording-boundary arithmetic, NOT a reconstruction of author RNG.

    The control cumulative-boundary array has N+1 entries: initial zero and one
    endpoint per recording. Published width=int(len(boundaries)*(1/folds)).
    Therefore positive per-file segment lengths cancel out when identifying
    which whole recordings occupy [fold*width, (fold+1)*width).
    No assumptions about the patient-branch offsets are made here.
    """
    _validate_controls(files, fold, folds)
    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)
    width = int((len(shuffled) + 1) * (1 / folds))
    start, stop = fold * width, (fold + 1) * width
    if width == 0:
        raise ValueError('published formula selects zero recordings for this input')
    if stop > len(shuffled):
        raise ValueError('published formula indexes beyond the control boundaries')
    return shuffled[:start] + shuffled[stop:], shuffled[start:stop]


def grouped_control_fold(files, seed=42, fold=0, folds=10):
    """Independent counterexample: shuffle unique IDs, allocate groups round-robin."""
    _validate_controls(files, fold, folds)
    subjects = sorted({subject_id(name) for name in files})
    if len(subjects) < folds:
        raise ValueError('grouped counterexample needs at least one subject per fold')
    random.Random(seed).shuffle(subjects)
    validation = set(subjects[fold::folds])
    return ([name for name in sorted(files) if subject_id(name) not in validation],
            [name for name in sorted(files) if subject_id(name) in validation])


def describe_split(train, validation):
    train_subjects = {subject_id(name) for name in train}
    val_subjects = {subject_id(name) for name in validation}
    overlap = sorted(train_subjects & val_subjects)
    return {
        'training_records': len(train), 'validation_records': len(validation),
        'training_subjects': len(train_subjects), 'validation_subjects': len(val_subjects),
        'training_filenames': train, 'validation_filenames': validation,
        'overlapping_subject_count': len(overlap), 'overlapping_subject_ids': overlap,
        'overlap_evidence': [
            {'subject_id': participant,
             'training': [name for name in train if subject_id(name) == participant],
             'validation': [name for name in validation if subject_id(name) == participant]}
            for participant in overlap
        ],
    }


def audit(records, seed=42, fold=0):
    counts = Counter(row['subject_id'] for row in records)
    controls = [row['filename'] for row in records if row['subject_id'][2:4] == 'Co']
    result = {
        'scope': 'control-branch filename/boundary counterexample; not training or historical reproduction',
        'audit_seed': seed, 'seed_is_original_author_seed': False, 'fold': fold,
        'ordering': 'lexicographic filenames, then fresh Python random.Random(seed).shuffle per subset',
        'assumption': 'each listed recording contributes positive segments; no waveforms loaded',
        'formula': 'width=int((N_control_records+1)*(1/10)); slice [fold*width:(fold+1)*width]',
        'sensor_files': len(records), 'participant_ids': len(counts),
        'participant_ids_with_multiple_recordings': sum(n > 1 for n in counts.values()),
        'records': records, 'control_replays': {},
    }
    for label, files in [('ElMaachi', controls),
                         ('Naimi_Ga_only', [name for name in controls if name.startswith('Ga')])]:
        train, validation = replay_control_fold(files, seed, fold)
        grouped_train, grouped_validation = grouped_control_fold(files, seed, fold)
        result['control_replays'][label] = {
            'input_control_records': len(files),
            'recording_boundary_replay': describe_split(train, validation),
            'grouped_counterexample': describe_split(grouped_train, grouped_validation),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='local checksum manifest; default is bundled official fixture')
    source.add_argument('--download-manifest', action='store_true',
                        help='explicitly fetch only the official manifest into memory')
    parser.add_argument('--seed', type=int, default=42, help='arbitrary audit seed, NOT original seed')
    parser.add_argument('--fold', type=int, choices=range(10), default=0)
    args = parser.parse_args(argv)
    try:
        records, provenance = load_manifest(args.manifest, args.download_manifest)
        result = audit(records, args.seed, args.fold)
        result['manifest'] = provenance
    except (OSError, ValueError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
