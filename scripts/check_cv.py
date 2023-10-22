from pathlib import Path
import polars as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--metric", type=str, default="valid/loss")
args = parser.parse_args()

metrics_csv_paths = list(Path(args.path).glob("*_metrics_fold*.csv"))

scores = []
for fp in metrics_csv_paths:
    print(fp)
    df = pl.read_csv(fp)
    print(df)
    scores.append(df[args.metric].min())

print(f"Scores: {scores}")
print(f"Mean: {sum(scores) / len(scores)}")
