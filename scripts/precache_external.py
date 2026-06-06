"""Pre-cache external source datasets to local parquet (run on a node WITH internet).

Polaris pins pyarrow<18 which conflicts with HuggingFace `datasets` (needs >=21). To keep
both worlds working we fetch the external source ONCE here (where polaris is importable),
persist it to parquet, and let the registry loader read that parquet at run time — so GPU
compute nodes never need polaris or internet.
"""
from __future__ import annotations

import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
BIOGEN_PARQUET = CACHE_DIR / "biogen_adme_fang_v1.parquet"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import polaris as po

    print(f"polaris {getattr(po, '__version__', '?')}")
    ds = po.load_dataset("biogen/adme-fang-v1")
    import pandas as pd

    tbl = ds.table if hasattr(ds, "table") else ds
    df = tbl if isinstance(tbl, pd.DataFrame) else tbl.to_pandas()
    print("loaded biogen/adme-fang-v1:", df.shape)
    print("columns:", list(df.columns))
    df.to_parquet(BIOGEN_PARQUET, index=False)
    print(f"wrote {BIOGEN_PARQUET} ({BIOGEN_PARQUET.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
