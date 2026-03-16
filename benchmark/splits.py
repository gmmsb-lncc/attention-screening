"""Scaffold split verification and generation.

Ensures pre-computed scaffold splits exist for the requested dataset(s)
before any level training begins.
"""

from __future__ import annotations

import os
import subprocess
import sys

from benchmark.config import BenchmarkConfig


def _exists_tsv_or_gz(path: str) -> str:
    """Check if a ``.tsv`` file exists, also trying the ``.tsv.gz`` variant.

    Returns the path that exists (preferring uncompressed), or an empty string.
    """
    if os.path.exists(path):
        return path
    gz = path + ".gz" if not path.endswith(".gz") else path
    if os.path.exists(gz):
        return gz
    return ""


def ensure_scaffold_splits(config: BenchmarkConfig) -> bool:
    """Verify or generate scaffold splits.

    All ``--dataset`` values now route through the **universal** split
    files (``universal_train.tsv``, ``universal_val.tsv``,
    ``universal_test.tsv``).  Returns ``True`` on success, ``False`` on
    failure.
    """
    scaffold_split_dir = config.scaffold_split_dir
    scenario_dir = os.path.join(scaffold_split_dir, "scenarios", "Sc")

    train = _exists_tsv_or_gz(os.path.join(scenario_dir, "universal_train.tsv"))
    val = _exists_tsv_or_gz(os.path.join(scenario_dir, "universal_val.tsv"))
    test = _exists_tsv_or_gz(os.path.join(scaffold_split_dir, "universal_test.tsv"))

    all_found = train and val and test

    if all_found and not config.force_split:
        print("  [OK] Universal scaffold splits found:")
        print(f"       train: {train}")
        print(f"       val:   {val}")
        print(f"       test:  {test}")
        if config.dataset != "all":
            print(f"       (will filter to dataset_source='{config.dataset}')")
        return True

    reason = "--force_split requested" if config.force_split else "missing split files"
    print(f"  [{reason}] Generating scaffold splits...")

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "scaffold_split.py"),
        "--output-dir",
        scaffold_split_dir,
        "--scenarios",
        "Sc",
    ]
    print(f"  Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: scripts/scaffold_split.py failed with code {exc.returncode}")
        return False
    except FileNotFoundError:
        print("  ERROR: scripts/scaffold_split.py not found")
        return False
