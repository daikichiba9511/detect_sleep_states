from torch.utils.data import DataLoader


# TODO: implement
class Runner:
    def __init__(self, is_val: bool = False) -> None:
        ...

    def _init_dl(self) -> DataLoader:
        ...

    def execute(self) -> None:
        ...


if __name__ == "__main__":
    runner = Runner(is_val=False)
    runner.execute()
