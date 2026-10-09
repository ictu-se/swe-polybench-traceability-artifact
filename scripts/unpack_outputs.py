"""Restore losslessly compressed experiment outputs and verify their checksums.

Run this after cloning the replication package, before analysis or resuming an
experiment. Existing matching outputs are retained; conflicting files are never
silently overwritten. Only destinations listed in the archived manifest are used.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from repository_benchmark import ROOT


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside_package(relative):
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError('Output path escapes the package: ' + relative)
    return path


def main():
    manifest = ROOT / 'data/compressed_outputs.json'
    if not manifest.exists():
        print('No compressed output manifest; experiment outputs are already unpacked.')
        return
    for record in json.loads(manifest.read_text()):
        source = inside_package(record['archive'])
        destination = inside_package(record['output'])
        if checksum(source) != record['archive_sha256']:
            raise ValueError('Compressed output checksum mismatch: ' + record['archive'])
        if destination.exists():
            if checksum(destination) != record['output_sha256']:
                raise ValueError('Existing output differs from the archive: ' + record['output'])
            print('Verified existing output:', record['output'])
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.unpacking')
        try:
            with gzip.open(source, 'rb') as reader, temporary.open('wb') as writer:
                shutil.copyfileobj(reader, writer)
            if (temporary.stat().st_size != record['output_bytes'] or
                    checksum(temporary) != record['output_sha256']):
                raise ValueError('Restored output checksum mismatch: ' + record['output'])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print('Restored and verified:', record['output'])


if __name__ == '__main__':
    main()
