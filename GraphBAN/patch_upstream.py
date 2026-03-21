#!/usr/bin/env python3
"""Patch upstream GraphBAN source for local reproducibility fixes.

Applies two safe/idempotent fixes:
1) Remove Colab-specific exit() pattern in models.py when IPythonConsole is missing.
2) Fix mutable graph reuse in inductive_mode/dataloader.py by deep-copying
   precomputed graph objects and adding a guard for negative virtual nodes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def patch_models(root: Path) -> list[str]:
    changed: list[str] = []
    targets = [
        root / "inductive_mode" / "models.py",
        root / "case_study" / "models.py",
    ]

    old = """try:
  from rdkit import Chem
  from rdkit.Chem.Draw import IPythonConsole
except ImportError:
  print('Stopping RUNTIME. Colaboratory will restart automatically. Please run again.')
  exit()"""

    new = """from rdkit import Chem
try:
  from rdkit.Chem.Draw import IPythonConsole
except ImportError:
  pass  # IPythonConsole only available in Jupyter/Colab"""

    for path in targets:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        if old in content:
            path.write_text(content.replace(old, new), encoding="utf-8")
            changed.append(str(path))

    return changed


def patch_dataloader(root: Path) -> list[str]:
    changed: list[str] = []
    path = root / "inductive_mode" / "dataloader.py"
    if not path.exists():
        return changed

    content = path.read_text(encoding="utf-8")
    original = content

    # Ensure copy import exists.
    if "import copy" not in content:
        if "import pandas as pd\n" in content:
            content = content.replace("import pandas as pd\n", "import pandas as pd\nimport copy\n", 1)
        else:
            content = "import copy\n" + content

    # Deep-copy cached graph object before mutating it.
    content = content.replace(
        "v_d = self.df.iloc[index]['drug']",
        "v_d = copy.deepcopy(self.df.iloc[index]['drug'])",
    )

    # Add guard against negative virtual nodes (if not already present).
    guard = (
        "        if num_virtual_nodes < 0:\n"
        "            raise ValueError(\n"
        "                f\"num_virtual_nodes={num_virtual_nodes} (max_drug_nodes={self.max_drug_nodes}, \"\n"
        "                f\"num_actual_nodes={num_actual_nodes}, index={index})\"\n"
        "            )\n"
    )

    if "num_virtual_nodes < 0" not in content:
        content = re.sub(
            r"(\s+num_virtual_nodes = self\.max_drug_nodes - num_actual_nodes\n)",
            r"\1" + guard,
            content,
        )

    if content != original:
        path.write_text(content, encoding="utf-8")
        changed.append(str(path))

    return changed


def patch_max_drug_nodes(root: Path, config_path: Path | None = None) -> list[str]:
    """Replace hardcoded max_drug_nodes=290 default with value from config YAML."""
    changed: list[str] = []
    path = root / "inductive_mode" / "dataloader.py"
    if not path.exists():
        return changed

    # Read MAX_NODES from config
    max_nodes = 290  # fallback
    if config_path is None:
        config_path = root.parent / "configs" / "kinase_inductive.yaml"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "MAX_NODES" in line and ":" in line:
                try:
                    max_nodes = int(line.split(":")[1].strip().split("#")[0].strip())
                except ValueError:
                    pass

    if max_nodes == 290:
        return changed  # nothing to patch

    content = path.read_text(encoding="utf-8")
    new_content = content.replace("max_drug_nodes=290", f"max_drug_nodes={max_nodes}")
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
        changed.append(f"{path} (max_drug_nodes=290 → {max_nodes})")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch upstream GraphBAN source with local fixes")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent / "src",
        help="Path to upstream GraphBAN source root (default: GraphBAN/src)",
    )
    args = parser.parse_args()

    src_root = args.src
    if not src_root.exists():
        raise FileNotFoundError(f"GraphBAN source path not found: {src_root}")

    changed = []
    changed.extend(patch_models(src_root))
    changed.extend(patch_dataloader(src_root))
    changed.extend(patch_max_drug_nodes(src_root))

    if changed:
        print("[INFO] Applied patches:")
        for c in changed:
            print(f"  - {c}")
    else:
        print("[INFO] No changes needed (already patched or patterns not found).")


if __name__ == "__main__":
    main()
