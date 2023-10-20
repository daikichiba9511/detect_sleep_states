import pprint
from pathlib import Path

import numpy as np
import polars as pl


class Config:
    root_dir: Path = Path(".")
    input_dir: Path = root_dir / "input"
    train_series_path: str | Path = (
        input_dir / "for_train" / "train_series_fold.parquet"
    )
    save_dir: Path = Path("./output/fe_exp000")
    num_array_save_fname: str = "num_array.npy"
    target_array_save_fname: str = "target_array.npy"
    mask_array_save_fname: str = "mask_array.npy"
    pred_use_array_save_fname: str = "pred_use_array.npy"
    series_ids_array_save_fname: str = "series_ids_array.npy"


num_array = np.load(Config.save_dir / Config.num_array_save_fname)
target_array = np.load(Config.save_dir / Config.target_array_save_fname)
mask_array = np.load(Config.save_dir / Config.mask_array_save_fname)
pred_use_array = np.load(Config.save_dir / Config.pred_use_array_save_fname)
series_ids_array = np.load(Config.save_dir / Config.series_ids_array_save_fname)

print(num_array.shape)
print(target_array.shape)
print(mask_array.shape)
print(pred_use_array.shape)
print(series_ids_array.shape)

print(series_ids_array[:10])
print(np.unique(series_ids_array).shape)

cnt = dict()
for uni_id in series_ids_array:
    if uni_id not in cnt:
        cnt[uni_id] = 0
    cnt[uni_id] += 1
pprint.pprint(cnt)


series = pl.read_parquet(Config.train_series_path)
label_cnt = dict()
for fold_i in range(5):
    valid_series = series.filter(pl.col("fold") == int(fold_i))
    print(f"Fold {fold_i}: {valid_series.shape[0]}")
    valid_series_ids = valid_series["series_id"].unique().to_list()
    print(len(valid_series_ids))

    # valid_series_idsに該当するもののindexのみを取得する
    # ためにvalid_series_idsに該当するマスク配列を作成
    # series_ids_array shape: (batch_size, )
    series_ids_mask = np.isin(series_ids_array, valid_series_ids)
    print(series_ids_mask.shape)
    print(np.unique(series_ids_array[series_ids_mask])[:10])
    print(np.unique(series_ids_array[series_ids_mask]).shape)

    num_array_ = num_array[series_ids_mask]
    target_array_ = target_array[series_ids_mask]

    print(num_array_.shape)
    print(target_array_.shape)

    print(num_array_[:10])
    print(target_array_[:10])
    label0 = target_array_[target_array_ == 0]
    label1 = target_array_[target_array_ == 1]
    label2 = target_array_[target_array_ == 2]
    print(label0.shape, label1.shape, label2.shape)
    label_cnt[fold_i] = (label0.shape[0], label1.shape[0], label2.shape[0])


print("Label Dist.", label_cnt)
