import json
import pathlib
import polars as pl

from src import utils

DATA_DIR = pathlib.Path("./input/child-mind-institute-detect-sleep-states")
FOR_TRAIN_DIR = pathlib.Path("./input/for_train")

series_df = pl.read_parquet(DATA_DIR / "train_series.parquet")
print(series_df)

periodic_dict = utils.create_periodic_dict(
    series_df.to_pandas(use_pyarrow_extension_array=True)
)
periodic_dict = {k: v.tolist() for k, v in periodic_dict.items()}
save_path = FOR_TRAIN_DIR / "periodic_dict.json"
with save_path.open("w") as f:
    json.dump(periodic_dict, f)
