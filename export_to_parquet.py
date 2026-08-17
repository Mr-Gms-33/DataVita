# -*- coding: utf-8 -*-
"""
Exports all tables from the local PostgreSQL database to compressed Parquet
files in ./data/, so the app can run without a live database connection
(e.g. on Streamlit Community Cloud, where a local Postgres isn't reachable).

Run this once locally whenever your source data changes:
    python export_to_parquet.py
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "Onlineshop")

# Tables to export. "events" is excluded by default (clickstream, very large,
# not required for the dashboards) ù remove from EXCLUDE if you want it too.
EXCLUDE_TABLES = set()

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

inspector = inspect(engine)
tables = [t for t in inspector.get_table_names() if t not in EXCLUDE_TABLES]

print(f"Found tables: {tables}")

# GitHub blocks any single file over 100 MB. Tables that would export
# above this threshold are automatically split into several smaller
# "<table>__partN.parquet" files, which the app's DuckDB fallback
# transparently reads back together as one table (see read_parquet glob
# in DataVita_V2.py).
MAX_FILE_MB = 80

total_size = 0
for table in tables:
    print(f"Exporting {table} ...")
    df = pd.read_sql_table(table, engine)

    # Quick size estimate to decide whether a split is needed.
    single_path = OUT_DIR / f"{table}.parquet"
    df.to_parquet(single_path, index=False, compression="snappy")
    size_mb = single_path.stat().st_size / (1024 * 1024)

    if size_mb <= MAX_FILE_MB:
        total_size += size_mb
        print(f"  -> {single_path.name}: {len(df):,} rows, {size_mb:.1f} MB")
        continue

    # Too big for one file -> remove the single file and split instead.
    single_path.unlink()
    n_parts = int(size_mb // MAX_FILE_MB) + 1
    chunk_size = -(-len(df) // n_parts)  # ceil division
    part_total = 0.0
    for i in range(n_parts):
        chunk = df.iloc[i * chunk_size: (i + 1) * chunk_size]
        if chunk.empty:
            continue
        part_path = OUT_DIR / f"{table}__part{i + 1}.parquet"
        chunk.to_parquet(part_path, index=False, compression="snappy")
        part_mb = part_path.stat().st_size / (1024 * 1024)
        part_total += part_mb
        print(f"  -> {part_path.name}: {len(chunk):,} rows, {part_mb:.1f} MB")
    total_size += part_total

print(f"\nDone. Total size: {total_size:.1f} MB in {OUT_DIR}")
if total_size > 90:
    print("WARNING: total size is close to/above GitHub's 100 MB per-file "
          "limit territory. Check individual file sizes above; consider "
          "excluding or sampling the largest table if any single file "
          "exceeds ~90 MB.")
