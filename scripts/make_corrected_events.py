import pathlib
import polars as pl

from src import utils

event_dir = pathlib.Path("./input/for_train/record_state.csv")
save_path = pathlib.Path(
    "./input/child-mind-institute-detect-sleep-states/train_events_corrected.csv"
)


# Preprocess to transform record_state.csv to train_events.csv form
record_state = pl.read_csv(event_dir)
transformed_record_state = utils.transformed_record_state(record_state)
print(transformed_record_state)
transformed_record_state.write_csv(save_path)
