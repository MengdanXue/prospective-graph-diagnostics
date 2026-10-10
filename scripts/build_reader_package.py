"""Build a flat scientific reader without author-specific execution state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    'scripts/reader_reproduce.py', 'scripts/reader_data.py', 'scripts/cpm_probe.py',
    'scripts/cpm_extension_worker.py', 'scripts/major_revision_audit.py',
    'experiments/prospective_data.py', 'experiments/prospective_models.py',
    'experiments/run_prospective_benchmark.py', 'configs/cpm_train_only_v1.json',
    'configs/reader_data_bindings_v1.json', 'configs/reader_mlp24_grid.json',
    'configs/prospective_benchmark_v2.json', 'requirements-reader.txt', 'LICENSE',
    'scripts/reviewer_source_sensitivity.py', 'scripts/cpm_partition_support.py',
    'scripts/export_reviewer_sensitivity.py', 'configs/reviewer_influence_source_groups_v1.json',
]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def build(evidence_base, cpm, output, review_analysis=None):
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(ROOT / 'docs/applied-intelligence/READER_REPRODUCTION.txt', output / 'README.txt')
    evidence = output / 'evidence'
    evidence.mkdir()
    shutil.copy2(cpm, evidence / 'cpm-extension.json')
    for folder, expected in [('input-retraining',1540),('mlp24',220)]:
        source = evidence_base / 'new-evidence' / folder / 'records'
        records = sorted(source.rglob('*.json'))
        if len(records) != expected:
            raise ValueError(f'incomplete {folder} records')
        for index, record in enumerate(records):
            # The JSON carries dataset/condition/seed/model identity. Flat short
            # names also make this package usable under Windows path limits.
            dest = evidence / 'new-evidence' / folder / 'records' / f'{index:06d}.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record, dest)
    if review_analysis is not None:
        target = evidence / 'review-sensitivity'
        target.mkdir()
        for name in ('published-metrics.json', 'SCOPE_BEFORE_EXECUTION.json',
                     'influence-source-analysis.json', 'partition-support.json',
                     'partition-support-public-webkb.json'):
            shutil.copy2(review_analysis / name, target / name)
        shutil.copy2(ROOT / 'output/major-revision-2026-10-08/major-revision-analysis.json',
                     target / 'matched-reference.json')
        shutil.copy2(ROOT / 'results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json',
                     target / 'primary-audit.json')
        for path in sorted((review_analysis/'tables').iterdir()):
            if path.suffix in ('.csv', '.json'):
                shutil.copy2(path, target / path.name)
    manifest = {'schema':'reader-package/1','files':{p.relative_to(output).as_posix():
        {'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(manifest['files']),'output':str(output)}))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-base',type=Path,required=True)
    p.add_argument('--cpm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-analysis',type=Path)
    a=p.parse_args()
    build(a.evidence_base.resolve(),a.cpm.resolve(),a.output.resolve(),
          a.review_analysis.resolve() if a.review_analysis else None)
