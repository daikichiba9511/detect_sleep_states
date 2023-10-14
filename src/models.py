from typing import Protocol

import torch
import torch.nn as nn


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
    def __init__(self, input_size, hidden_size, out_size, n_layers, bidir=True):
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

    def forward(self, x, h=None):
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
        #         x = F.normalize(x,dim=0)
        return x, new_h  # log probabilities + hidden states


class ModelConfig(Protocol):
    model_type: str
    input_size: int
    hidden_size: int
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


if __name__ == "__main__":
    test_run_model()
