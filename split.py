import argparse

import duckdb

p = argparse.ArgumentParser()
p.add_argument("--input", default="Benchmark/MTQ-Auto/test.parquet")
p.add_argument("--source", help="data_source value, e.g. er, hca, ct, wwc")
p.add_argument("--out", help="output parquet (default: test-<source>.parquet)")
p.add_argument("--list", action="store_true")
args = p.parse_args()

if args.list or not args.source:
    duckdb.sql(
        f"SELECT data_source, count(*) AS n FROM read_parquet('{args.input}') "
        f"GROUP BY 1 ORDER BY 2 DESC"
    ).show()
else:
    out = args.out or f"Benchmark/MTQ-Auto/test-{args.source}.parquet"
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{args.input}') "
        f"WHERE data_source = '{args.source}') "
        f"TO '{out}' (FORMAT 'PARQUET')"
    )
    print(f"Created {out}")
