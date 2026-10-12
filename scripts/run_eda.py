#!/usr/bin/env python3
"""Chay toan bo EDA va xuat bang CSV + bieu do vao `reports/`.

Chay:
    python scripts/run_eda.py
    python scripts/run_eda.py --synthetic     # khi chua tai du lieu goc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, eda  # noqa: E402
from src.data_loader import load_raw_data, make_synthetic_dataset, profile_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Dung du lieu gia lap.")
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "eda"))
    args = parser.parse_args(argv)

    df = make_synthetic_dataset(2000) if args.synthetic else load_raw_data()
    print(f"Profile: {profile_dataset(df)}\n")

    tables = eda.run_full_eda(df, output_dir=args.output)
    for name, table in tables.items():
        print(f"--- {name} ---")
        print(table.to_string(index=False))
        print()

    figure_path = config.FIGURES_DIR / "eda_overview.png"
    eda.plot_eda_overview(df, save_path=figure_path)
    print(f"Da luu bieu do: {figure_path}")
    print(f"Da luu bang CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
