"""Download the exact CodeRankEmbed revision used by the semantic controls."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from repository_benchmark import ROOT
from semantic_rerank import CODE_MODEL,CODE_REVISION


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    snapshot_download(CODE_MODEL,revision=CODE_REVISION,local_dir=args.output)
    manifest=json.loads((ROOT/'results/extension/semantic_environment.json').read_text())['model_asset_sha256']
    for name,digest in manifest.items():
        assert hashlib.sha256((args.output/name).read_bytes()).hexdigest()==digest,name
    print('Pinned encoder downloaded and all archived asset hashes verified')

if __name__=='__main__':main()
