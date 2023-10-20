from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(hidden_size * dir_factor, hidden_size * dir_factor * 2)
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h


class MultiResidualBiGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        out_size: int,
        n_layers: int,
        bidir: bool = True,
    ) -> None:
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor, h=None) -> tuple[torch.Tensor, list]:
        # if we are at the beginning of a sequence (no hidden state)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        x = self.fc_out(x)
        # x = F.normalize(x, dim=0)  # 系列の中でどのtime_stepがonset,wakeupそれぞれ一番確率が大きいか
        return x, new_h  # log probabilities + hidden states


class SleepRNNMocel(nn.Module):
    """
    Refs:
    [1] https://github.com/TakoiHirokazu/Kaggle-Parkinsons-Freezing-of-Gait-Prediction/blob/main/takoi/exp/ex143_tdcsfog_gru.ipynb
    """

    def __init__(
        self,
        dropout: float = 0.2,
        input_num_size: int = 12,
        num_linear_size: int = 64,
        model_size=128,
        linear_out=128,
        out_size: int = 3,
    ) -> None:
        super().__init__()

        self.num_linear = nn.Sequential(
            nn.Linear(input_num_size, num_linear_size),
            nn.LayerNorm(num_linear_size),
        )

        self.rnn = nn.GRU(
            num_linear_size,
            model_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc_out = nn.Sequential(
            nn.Linear(model_size * 2, linear_out),
            nn.LayerNorm(linear_out),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_out, out_size),
        )

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "rnn" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

    def forward(
        self,
        num_array: torch.Tensor,
        mask_array: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_embed = self.num_linear(num_array)
        out, _ = self.rnn(num_embed)
        out = self.fc_out(out)
        return out


class ModelConfig(Protocol):
    model_type: str
    input_size: int
    hidden_size: int
    model_size: int
    linear_out: int
    out_size: int
    n_layers: int
    bidir: bool


def build_model(config: ModelConfig) -> torch.nn.Module:
    if config.model_type == "MultiResidualBiGRU":
        model = MultiResidualBiGRU(
            config.input_size,
            config.hidden_size,
            config.out_size,
            config.n_layers,
            bidir=config.bidir,
        )
        return model
    elif config.model_type == "SleepRNNModel":
        model = SleepRNNMocel(
            input_num_size=config.input_size,
            num_linear_size=config.hidden_size,
            model_size=config.model_size,
            linear_out=config.linear_out,
            out_size=config.out_size,
        )
        return model

    else:
        raise NotImplementedError


def test_run_model():
    model = MultiResidualBiGRU(input_size=10, hidden_size=64, out_size=2, n_layers=5)
    model = model.train()

    max_chunk_size = 10
    bs = 32
    x = torch.randn(bs, max_chunk_size)
    h = None
    p, h = model(x, h)
    print("pred: ", p.shape)
    print(len(h))
    print([h_i.shape for h_i in h])


def test_run_model2():
    num_feats = 8

    model = SleepRNNMocel(input_num_size=num_feats).cuda()
    model = model.train()

    seq_len = 1000  # ここは自由
    # seq_len = 10000
    bs = 8 * 190  # max bs
    num_arr = torch.randn(bs, seq_len, num_feats).cuda()
    mask_arr = torch.ones(bs, seq_len).cuda()
    attention_mask = torch.ones(bs, seq_len).cuda()
    y = torch.randint(0, 1, (bs, seq_len, 3)).float().cuda()
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    print("num_arr: ", num_arr.shape)
    print("mask_arr: ", mask_arr.shape)
    print("attention_mask: ", attention_mask.shape)
    # p: (bs, seq_len, out_size)
    with torch.cuda.amp.autocast_mode.autocast(dtype=torch.bfloat16):
        p = model(num_arr, mask_arr, attention_mask)
        loss = criterion(p, y)
        scaler.scale(loss).backward()  # type: ignore

    print("pred: ", p.shape)


if __name__ == "__main__":
    # test_run_model()
    test_run_model2()
