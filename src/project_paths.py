"""Where things live on Google Drive (used by the notebooks on the Colab runtime).

Layout (MyDrive/rs_project/):
  datasets/rs_dataset_256.zip      generated dataset (+ datasets/<name>_config.json)
  runs/<YYYY-MM-DD>_<name>/        one folder per training run:
                                   best.pt, last.pt, log.csv, run_info.json,
                                   results.json, eval.json, preview_epXX.png
  results/<run>/                   inference outputs (infer.py)
  .github_token                    optional: GitHub token for cloning the repo

Older runs (before 2026-10-24) were in MyDrive/rs_runs/<name> and the dataset in
MyDrive/rs_dataset_256.zip; `migrate_old_layout()` copies them into the new
layout (the originals are left untouched).
"""
import shutil
from pathlib import Path

MYDRIVE = Path("/content/drive/MyDrive")
ROOT = MYDRIVE / "rs_project"
DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

OLD_RUNS = MYDRIVE / "rs_runs"
OLD_DATASET_ZIP = MYDRIVE / "rs_dataset_256.zip"
OLD_RUN_DATE = "2026-10-13"          # stage-1 runs were trained on this day


def ensure_dirs():
    for d in (DATASETS, RUNS, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


def migrate_old_layout(verbose=True):
    """Copy old runs (rs_runs/<name>) to runs/2026-10-13_<name>. Small files only;
    the dataset zip is not copied (find_dataset_zip() looks in both places)."""
    ensure_dirs()
    if not OLD_RUNS.exists():
        return
    for old in sorted(p for p in OLD_RUNS.iterdir() if p.is_dir()):
        new = RUNS / f"{OLD_RUN_DATE}_{old.name}"
        if new.exists():
            continue
        shutil.copytree(old, new)
        if verbose:
            print(f"copied {old} -> {new}")


def find_dataset_zip(name="rs_dataset_256"):
    for p in (DATASETS / f"{name}.zip", MYDRIVE / f"{name}.zip"):
        if p.exists():
            return p
    return None


def find_run(name, file="best.pt"):
    """Newest run whose folder name ends with `name` (e.g. 'motion_dual')."""
    cands = sorted(RUNS.glob(f"*_{name}/{file}")) + sorted(RUNS.glob(f"{name}/{file}"))
    if not cands and (OLD_RUNS / name / file).exists():
        cands = [OLD_RUNS / name / file]
    return cands[-1] if cands else None


def status():
    """Print what is available on Drive."""
    ensure_dirs()
    z = find_dataset_zip()
    print("dataset zip :", f"{z} ({z.stat().st_size / 1e9:.2f} GB)" if z else "BRAK - uruchom 01_dataset.ipynb")
    runs = sorted(p for p in RUNS.iterdir() if p.is_dir())
    print(f"runs ({RUNS}):")
    for r in runs:
        files = {f.name for f in r.iterdir()}
        print(f"  {r.name:32s} best.pt: {'tak' if 'best.pt' in files else 'NIE'}"
              f" | results.json: {'tak' if 'results.json' in files else '-'}")
    if not runs:
        print("  (brak)")
