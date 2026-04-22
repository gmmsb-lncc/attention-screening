"""Inspeciona features_extracted.pkl do GraphBAN.

Uso em diamante-02:
    cd /data/docktkinase
    python inspect_graphban_cache.py
"""

import pickle
from pathlib import Path

REPO = Path(__file__).parent.resolve()

CACHES = [
    REPO / "GraphBAN/results/non_human/feature_cache/features_extracted.pkl",
    REPO / "GraphBAN/results/human/feature_cache/features_extracted.pkl",
    REPO / "GraphBAN/results/all/feature_cache/features_extracted.pkl",
]


def inspect(p: Path) -> None:
    print(f"\n=== {p} ===")
    if not p.exists():
        print("  (não existe)")
        return
    d = pickle.load(open(p, "rb"))
    print(f"  top-level type: {type(d).__name__}")
    if isinstance(d, dict):
        keys = list(d.keys())
        print(f"  n keys: {len(keys)}")
        print(f"  first 5 keys: {keys[:5]}")
        k = keys[0]
        v = d[k]
        print(f"  value[0] type: {type(v).__name__}")
        if hasattr(v, "shape"):
            print(f"  value[0] shape: {v.shape}")
            print(f"  value[0] dtype: {v.dtype}")
        elif hasattr(v, "__len__"):
            print(f"  value[0] len: {len(v)}")
        if isinstance(v, dict):
            print(f"  value[0] sub-keys: {list(v.keys())[:10]}")
    elif hasattr(d, "columns"):
        print(f"  DataFrame columns: {list(d.columns)}")
        print(f"  n rows: {len(d)}")
        print(f"  first row types: {d.dtypes.to_dict()}")
        row = d.iloc[0]
        for col in d.columns:
            val = row[col]
            if hasattr(val, "shape"):
                print(f"    {col}: {type(val).__name__} shape={val.shape}")
            elif isinstance(val, (str, int, float)):
                s = str(val)[:80]
                print(f"    {col}: {type(val).__name__} = {s!r}")
            elif hasattr(val, "__len__"):
                print(f"    {col}: {type(val).__name__} len={len(val)}")
            else:
                print(f"    {col}: {type(val).__name__}")
    elif isinstance(d, (list, tuple)):
        print(f"  len: {len(d)}")
        print(f"  item[0] type: {type(d[0]).__name__}")


def main() -> None:
    for p in CACHES:
        inspect(p)


if __name__ == "__main__":
    main()
