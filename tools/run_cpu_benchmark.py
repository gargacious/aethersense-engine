#!/usr/bin/env python3
"""Build/run the fixed CPU workload locally; no downloads or privileged setup."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error('--runs must be between 1 and 10')
    root = Path(__file__).resolve().parents[1]
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic', '-Werror']
    sources = ['src/engine.cpp', 'src/dsp_extensions.cpp', 'apps/benchmark.cpp']
    compiler = subprocess.run([args.compiler, '--version'], check=True, text=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory(prefix='aethersense-cpu-benchmark-') as directory:
        executable = Path(directory) / 'cpu-benchmark'
        command = [args.compiler, *flags, '-I' + str(root / 'include'), '-I' + str(root / 'third_party'),
                   *(str(root / source) for source in sources), '-o', str(executable)]
        subprocess.run(command, check=True)
        runs = []
        for _ in range(args.runs):
            result = subprocess.run([str(executable)], check=True, text=True, capture_output=True)
            runs.append(json.loads(result.stdout))
    try:
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                                  text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError:
        revision = 'unavailable'
    document = {'schema_version': 1, 'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'environment': {'platform': platform.platform(), 'processor': platform.processor(),
                                'python': platform.python_version(), 'compiler': compiler},
                'source_revision': revision, 'source_state': 'content hashes identify possible uncommitted sources',
                'build_flags': flags, 'source_sha256': {
                    source: hashlib.sha256((root / source).read_bytes()).hexdigest() for source in sources},
                'runs': runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + '\n')
    for index, run in enumerate(runs, 1):
        print('run', index, 'input', run['input_fingerprint'])
        for name, metric in run['measurements'].items():
            print(f"  {name}: p50={metric['p50_us']:.3f} us p99={metric['p99_us']:.3f} us "
                  f"throughput={metric['throughput_items_per_s']:.1f} items/s")
    print('saved', args.output.resolve())

if __name__ == '__main__':
    main()
