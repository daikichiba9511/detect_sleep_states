from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBiGRU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_layers: int = 1,
        bidir: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout,
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
    """ResidualBiGRUを複数重ねた系列モデル

    Attributes:
        input_size: input feature size
        hidden_size: hidden feature size
        out_size: output feature size
        n_layers: number of layers of ResidualBiGRU

        fc_in: (bs, seq_len, input_size) -> (bs, seq_len, hidden_size)
        ln: LayerNorm
        res_bigrus: list of ResidualBiGRU
        fc_out: (bs, seq_len, hidden_size) -> (bs, seq_len, out_size)

    References:
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        out_size: int,
        n_layers: int,
        bidir: bool = True,
        dropuot: float = 0.0,
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
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir, dropout=dropuot)
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


def create_conv1dnn(in_chans: int, out_chans: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_chans, out_chans, kernel_size, padding=kernel_size // 2),
        nn.ReLU(),
    )


class MultiResidualBiGRUMultiKSConv1D(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        out_size: int,
        n_layers: int,
        bidir: bool = True,
        gru_n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.mlp_in = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.conv1d_ks3 = create_conv1dnn(hidden_size, hidden_size, 3)
        self.conv1d_ks7 = create_conv1dnn(hidden_size, hidden_size, 7)
        self.conv1d_ks12 = create_conv1dnn(hidden_size, hidden_size, 11)

        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(
                    hidden_size * 4, n_layers=gru_n_layers, bidir=bidir, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size * 4, out_size)

    def forward(self, x: torch.Tensor, h=None) -> tuple[torch.Tensor, list]:
        # if we are at the beginning of a sequence (no hidden state)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        # (BS, seq_len, n_feats) -> (BS, seq_len, hidden_size)
        x = self.mlp_in(x)
        # (BS, seq_len, hidden_size) -> (BS, hidden_size, seq_len)
        x = x.transpose(1, 2)
        # x: (bs, seq_len * 3, n_feats)
        x = torch.cat(
            [
                self.conv1d_ks3(x),
                self.conv1d_ks7(x),
                self.conv1d_ks12(x),
                x,
            ],
            dim=1,
        )
        x = x.transpose(1, 2)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        # (BS, seq_len, n_feats) -> (BS, seq_len, out_size)
        x = self.fc_out(x)
        return x, new_h  # log probabilities + hidden states


class TransformerEncoderLayer(nn.Module):
    """

    Attributes:
        mha: MultiheadAttention
        ln1: LayerNorm
        ln2: LayerNorm
        seq: nn.Sequential

    Refs:
    [1]
    https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-tdcsfog-training-code
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        seq_model_dim: int = 320,
        encoder_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.seq = nn.Sequential(
            nn.Linear(seq_model_dim, seq_model_dim),
            nn.ReLU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(seq_model_dim, seq_model_dim),
            nn.Dropout(encoder_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, attn_weights = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        x = x + self.seq(x)
        x = self.ln2(x)
        return x


class SleepTransformerEncoder(nn.Module):
    """

    Refs:
    [1]
    https://www.kaggle.com/code/baurzhanurazalinov/parkinson-s-freezing-tdcsfog-training-code
    """

    def __init__(
        self,
        model_dim: int = 320,
        dropout_rate: float = 0.2,
        num_encoder_layers: int = 3,
        num_lstm_layers: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        seq_model_dim: int = 320,
        seq_len: int = 3000,
    ):
        super().__init__()
        self.first_linear = nn.Linear(model_dim, model_dim)
        self.first_dropout = nn.Dropout(dropout_rate)
        self.encoder_layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                seq_model_dim=seq_model_dim,
                encoder_dropout=dropout_rate,
            )
            for _ in range(num_encoder_layers)
        ]
        self.lstm_layers = [
            nn.LSTM(model_dim, model_dim, bidirectional=True, batch_first=True)
            for _ in range(num_lstm_layers)
        ]
        self.seq_len = seq_len
        self.pos_encoding = nn.Parameter()


class SleepRNNMocel(nn.Module):
    """

    Attributes:
        num_linear: (bs, seq_len, num_feats) -> (bs, seq_len, num_linear_size)

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
    seq_len: int


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
    elif config.model_type == "MultiResidualBiGRUMultiKSConv1D":
        model = MultiResidualBiGRUMultiKSConv1D(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            out_size=config.out_size,
            n_layers=config.n_layers,
            bidir=config.bidir,
        )
        return model

    else:
        raise NotImplementedError


def _test_run_model():
    print("Test1")
    model = MultiResidualBiGRU(input_size=10, hidden_size=64, out_size=2, n_layers=5)
    model = model.train()

    max_chunk_size = 10
    bs = 32
    seq_len = 1000
    x = torch.randn(bs, seq_len, max_chunk_size)
    h = None
    p, h = model(x, h)
    print("pred: ", p.shape)
    print(len(h))
    print([h_i.shape for h_i in h])


def _test_run_model2():
    print("Test2")
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


def _test_run_model3():
    print("Test3")
    model = MultiResidualBiGRUMultiKSConv1D(
        input_size=10, hidden_size=256, out_size=2, n_layers=5
    )
    model = model.train()

    max_chunk_size = 10
    seq_len = 3000
    bs = 8 * 2
    x = torch.randn(bs, seq_len, max_chunk_size)
    print(x.shape)
    h = None

    profile = False
    if profile:
        import os
        import memray

        os.remove("./model_forwad.bin")
        with memray.Tracker("model_forwad.bin"):
            p, h = model(x, h)
            print("pred: ", p.shape)
            print(len(h))
            print([h_i.shape for h_i in h])
    else:
        p, h = model(x, h)
        print("pred: ", p.shape)
        print(len(h))
        print([h_i.shape for h_i in h])


if __name__ == "__main__":
    # _test_run_model()
    # _test_run_model2()
    _test_run_model3()
