"""Read ZIP/pickle opcode metadata without unpickling or importing saved globals."""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickletools
import sys
import zipfile


MAX_PICKLE_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 10000
MAX_OPCODES = 200000


def inspect_checkpoint(path, max_pickle_bytes=MAX_PICKLE_BYTES):
    """Return lexical metadata, not reconstructed objects or forward semantics.

    No archive extraction, pickle.load, torch.load, eval, or saved-global imports.
    GLOBAL names are literal tokens. STACK_GLOBAL is counted but NOT resolved.
    Size/opcode limits reduce resource use; this is not a malware certification.
    """
    if not isinstance(max_pickle_bytes, int) or max_pickle_bytes <= 0:
        raise ValueError('max_pickle_bytes must be a positive integer')
    try:
        with zipfile.ZipFile(path, 'r') as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ENTRIES:
                raise ValueError('archive has too many members')
            candidates = [entry for entry in entries
                          if entry.filename == 'data.pkl' or entry.filename.endswith('/data.pkl')]
            if len(candidates) != 1:
                raise ValueError('expected exactly one data.pkl member in a ZIP checkpoint')
            entry = candidates[0]
            if entry.flag_bits & 1:
                raise ValueError('encrypted pickle members are unsupported')
            if entry.file_size > max_pickle_bytes:
                raise ValueError('pickle metadata exceeds size limit')
            with archive.open(entry) as handle:
                payload = handle.read(max_pickle_bytes + 1)
            if len(payload) > max_pickle_bytes:
                raise ValueError('pickle metadata exceeds size limit')
            members = [{'name': item.filename, 'bytes': item.file_size,
                        'compressed_bytes': item.compress_size} for item in entries]
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError('cannot read checkpoint as an unencrypted ZIP archive') from error

    globals_seen = []
    strings = []
    integer_tokens = []
    opcodes = Counter()
    stop_position = None
    try:
        for index, (opcode, value, position) in enumerate(pickletools.genops(payload)):
            if index >= MAX_OPCODES:
                raise ValueError('pickle exceeds opcode limit')
            opcodes[opcode.name] += 1
            if opcode.name == 'GLOBAL':
                globals_seen.append({'offset': position, 'token': value})
            elif opcode.name in {'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
                                 'STRING', 'BINSTRING', 'SHORT_BINSTRING'}:
                # Bound output as well as input. Bytes are represented, never decoded/executed.
                text = value if isinstance(value, str) else repr(value)
                strings.append({'offset': position, 'token': text[:200],
                                'truncated': len(text) > 200})
            elif opcode.name in {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4'}:
                integer_tokens.append({'offset': position, 'value': value})
            if opcode.name == 'STOP':
                stop_position = position
    except (ValueError, UnicodeError, OverflowError) as error:
        raise ValueError(f'invalid or excessive pickle opcode stream: {error}') from error
    if stop_position is None or stop_position != len(payload) - 1:
        raise ValueError('pickle must end with STOP and contain no trailing payload')
    return {
        'format': 'ZIP checkpoint; static opcode inspection only',
        'pickle_executed': False, 'pickle_bytes': len(payload),
        'archive_members': members, 'opcode_counts': dict(sorted(opcodes.items())),
        'global_tokens': globals_seen,
        'unresolved_stack_globals': opcodes['STACK_GLOBAL'],
        'string_tokens': strings, 'integer_tokens': integer_tokens,
        'limitations': [
            'Tokens are not reconstructed module attributes or validated tensor shapes.',
            'Forward/residual/pooling behavior and training provenance cannot be inferred.',
            'Token output may expose saved names or paths; review before sharing.',
            'No tensor contents are read; legacy non-ZIP serialization is unsupported.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_checkpoint(args.checkpoint)
    except ValueError as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
