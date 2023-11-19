from typing import Any, Callable, Protocol

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TAT
import torchvision.transforms.functional as TF

from src import augmentations, decoders, encoders, feature_extractors


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
            # -- epx018
            # nn.Linear(input_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
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
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=encoder_dropout,
            device=device,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(embed_dim, device=device)
        self.seq = nn.Sequential(
            nn.Linear(seq_model_dim, seq_model_dim),
            nn.ReLU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(seq_model_dim, seq_model_dim),
            nn.Dropout(encoder_dropout),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Enc 1: x.shape", x.shape)
        attn_out, _ = self.mha(query=x, value=x, key=x, need_weights=False)
        x = self.ln(x + attn_out)
        # print("Enc 2: x.shape", x.shape)
        x = x + self.seq(x)
        # print("Enc 3: x.shape", x.shape)
        x = self.ln(x)
        # print("Enc 4: x.shape", x.shape)
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
        device: torch.device = torch.device("cuda"),
        bs: int = 24,
    ):
        super().__init__()
        self.first_linear = nn.Linear(model_dim, embed_dim)
        self.first_dropout = nn.Dropout(dropout_rate)
        self.encoder_layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                seq_model_dim=seq_model_dim,
                encoder_dropout=dropout_rate,
                device=device,
            )
            for _ in range(num_encoder_layers)
        ]
        self.rnn_layers = [
            nn.LSTM(embed_dim, embed_dim // 2, bidirectional=True, batch_first=True).to(
                device
            )
            # nn.GRU(embed_dim, embed_dim // 2, bidirectional=True, batch_first=True).to(
            #     device
            # )
            for _ in range(num_lstm_layers)
        ]
        self.seq_len = seq_len
        data = torch.normal(mean=0, std=0.02, size=(1, self.seq_len, embed_dim))
        self.pos_encoding = nn.Parameter(
            data.to(device),
            requires_grad=True,
        )
        self.bs = bs

    def forward(
        self, x: torch.Tensor, training: bool = False, bs: int = 24
    ) -> torch.Tensor:
        """
        Args:
            x: (bs, seq_len, model_dim)

        Returns:
            x: (bs, seq_len, model_dim)
        """
        x = self.first_linear(x)
        # if training:
        if False:
            bs = self.bs
            # augmentation by randomly roll of the position encoding tensor
            tile = torch.tile(self.pos_encoding, dims=(bs, 1, 1))
            shifts = tuple(
                map(int, torch.randint(low=-self.seq_len, high=0, size=(bs,)))
            )
            random_pos_encoding = torch.roll(tile, shifts=shifts, dims=[1] * bs)
            # print(f"{x.shape=}, {random_pos_encoding.shape=}")
            x = x + random_pos_encoding
        else:
            bs = x.shape[0]
            # print(f"{x.shape=}, {self.pos_encoding.shape=}, {self.pos_encoding=}")
            tile = torch.tile(input=self.pos_encoding, dims=(bs, 1, 1))
            x = x + tile

        x = self.first_dropout(x)

        # print(f"{x.shape =}")
        # Ref:
        # [1]
        # https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-112/notebook#Build-Model
        for i in range(len(self.encoder_layers)):
            x_old = x
            x = self.encoder_layers[i](x)
            x = 0.7 * x + 0.3 * x_old  # skip connrection
            # print(f"{i}: {x.shape =}")

        for i in range(len(self.rnn_layers)):
            # print(f"Lstm: 1 {i}: {x.shape =}")
            x, _ = self.rnn_layers[i](x)
            # print(f"Lstm: 2 {i}: {x.shape =}")

        return x


class SleepTransformer(nn.Module):
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
        out_dim: int = 2,
        device: torch.device = torch.device("cuda"),
        fc_hidden_size: int = 128,
        bs: int = 24,
    ):
        super().__init__()
        self.encoder = SleepTransformerEncoder(
            model_dim=model_dim,
            dropout_rate=dropout_rate,
            num_encoder_layers=num_encoder_layers,
            num_lstm_layers=num_lstm_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            seq_model_dim=seq_model_dim,
            device=device,
            bs=bs,
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_size, out_dim),
        )

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self.encoder(x, training=training)
        x = self.fc(x)
        return x


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


def made_sample_weights(labels: torch.Tensor) -> torch.Tensor:
    num_zero_label = (labels == 0.0).sum().float()
    sample_weights = labels.sum(dim=1, keepdim=False) / labels.shape[1]
    sample_weights[sample_weights == 0.0] = num_zero_label
    sample_weights = 1 / sample_weights
    return sample_weights


def weighted_loss(loss: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    """Weighted loss function

    Args:
        loss: (batch_size, pred_len, n_classes)
        sample_weights: (batch_size, n_classes)

    Returns:
        loss: 1-dim weighted loss that shape is (1, )
    """
    return (loss.mean(1) * sample_weights).mean()


class MelConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv1 = nn.Conv2d(
            in_channels=in_chans,  # type: ignore
            out_channels=out_chans,  # type: ignore
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_chans,  # type: ignore
            out_channels=out_chans,  # type: ignore
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        return x


class AuxHead(nn.Module):
    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_feats, out_feats)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # (batch_size, pred_len, n_classes) -> (batch_size, n_classes, pred_len)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        loss = self.loss_fn(x, label)
        return loss


class Spectrogram2DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: str | None = None,
        use_sample_weights: bool = False,
        use_aux_head: bool = False,
        use_custom_encoder: bool = True,
        spec_augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        # exp020~049まで
        if use_custom_encoder:
            self.encoder = encoders.CustomUnet(
                name=encoder_name,
                pretrained=encoder_weights is not None,
                decoder_channels=[258, 128, 64, 32, 16],
                n_classes=1,
                # dropout=0.2,
            )
        else:
            self.encoder = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
            )
        self.decoder = decoder
        self.use_sample_weights = use_sample_weights
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none" if use_sample_weights else "mean"
        )
        self.spec_augment = spec_augment
        self.use_aux_head = use_aux_head
        if self.use_aux_head:
            self.aux_head = AuxHead(in_feats=5760, out_feats=1)

        # self.mel_spectrogram = nn.Sequential(
        #     # forward: (..., times) -> (n_channels, n_mels, times)
        #     TAT.MelSpectrogram(
        #         sample_rate=16000,
        #         n_fft=400,  # size of fft,creates n_fft//2+1 bins
        #         n_mels=64,  # number of mel filterbanks
        #         win_length=100,  # defaults None is equal to n_fft
        #         hop_length=None,  # defaults None is equal to win_length//2
        #     ),
        # )
        # self.mel_conv = MelConvBlock(
        #     in_chans=feature_extractor.in_channels,  # type: ignore
        #     out_chans=feature_extractor.out_chans,  # type: ignore
        # )
        # self.mel_fc = nn.Linear(
        #     in_features=116,  # type: ignore mel_bins
        #     out_features=2880,  # type: ignore seq_len//downsample_rate
        # )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        do_mixup: bool = False,
        do_mixup_raw_signal: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        if do_mixup_raw_signal and labels is not None:
            x, labels, mixed_labels, lam = augmentations.mixup(x, labels)
        x1 = self.feature_extractor(x)  # (bs,nc,height,seq_len//downsample_rate)
        # print("wavegram", x1.shape)

        # (bs,n_feats,seq_len) -> (bs,feature_extractor.out_chans,n_mels,mel_bins)
        # x = self.mel_spectrogram(x)
        # print("mel_spectrogram", x.shape, "mel_spectrogram", x.shape)
        # if self.spec_augment is not None:
        #     x = self.spec_augment(x)
        #
        if do_mixup and labels is not None and not do_mixup_raw_signal:
            # x, labels, mixed_labels, lam = augmentations.mixup(x, labels)
            x1, labels, mixed_labels, lam = augmentations.mixup(x1, labels)
        else:
            mixed_labels, lam = None, None

        if do_cutmix and labels is not None:
            x1, labels, _, _ = augmentations.cutmix(x1, labels)
        #
        # print("mel_conv before", x.shape)
        # x = self.mel_conv(x)
        # print(f"mel_spectrogram.shape: {x.shape}, wavegram.shape: {x1.shape}")
        x1 = self.encoder(x1).squeeze(1)  # (batch_size, height, seq_len)

        logits = self.decoder(x1)  # (batch_size, n_classes, n_timesteps)
        output = {"logits": logits}
        if labels is not None:
            # (batch_size, pred_len, n_classes)
            if sample_weights is None:
                loss = self.loss_fn(logits, labels)
            else:
                loff_fn = nn.BCEWithLogitsLoss(reduction="none")
                # (batch_size, pred_len, n_classes)
                loss = loff_fn(logits, labels)
                # (batch_size, pred_len)
                loss = torch.mean(sample_weights * loss.mean(1).mean(1))

            if self.use_aux_head:
                aux_labels = torch.max(labels, dim=1)[0].unsqueeze(-1)
                aux_loss = self.aux_head(logits, aux_labels)
                loss = loss + aux_loss

            if self.use_sample_weights:
                # label \in [0.0, 1.0]
                # make sample_weights for each class
                sample_weights = made_sample_weights(labels)
                loss = weighted_loss(loss, sample_weights)
            #
            if do_mixup and mixed_labels is not None and lam is not None:
                mixed_loss = self.loss_fn(logits, mixed_labels)
                if self.use_sample_weights:
                    sample_weights = made_sample_weights(mixed_labels)
                    mixed_loss = weighted_loss(mixed_loss, sample_weights)

                loss = lam * loss + (1 - lam) * mixed_loss
                # loss = mixed_loss

            if do_cutmix and labels is not None:
                cutmix_loss = self.loss_fn(logits, labels)
                loss = cutmix_loss

            output["loss"] = loss
        return output


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
    batch_size: int

    train_seq_len: int
    transformer_params: dict[str, Any] | None
    spectrogram2dcnn_params: dict[str, Any] | None
    downsample_rate: int

    use_spec_augment: bool
    spec_augment_params: dict[str, Any] | None


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
    elif config.model_type == "SleepTransformer":
        if config.transformer_params is None:
            raise ValueError("config.transformer_params is None")

        model = SleepTransformer(
            model_dim=config.transformer_params["model_dim"],
            dropout_rate=config.transformer_params["dropout"],
            num_encoder_layers=config.transformer_params["num_encoder_layers"],
            num_lstm_layers=config.transformer_params["num_lstm_layers"],
            embed_dim=config.transformer_params["embed_dim"],
            num_heads=config.transformer_params["num_heads"],
            seq_model_dim=config.transformer_params["seq_model_dim"],
            seq_len=config.transformer_params["seq_len"],
            out_dim=config.out_size,
            fc_hidden_size=config.transformer_params["fc_hidden_dim"],
            bs=config.batch_size,
        )
        return model

    elif config.model_type == "Spectrogram2DCNN":
        if config.spectrogram2dcnn_params is None:
            raise ValueError("config.spectrogram2dcnn_params is None")
        params = config.spectrogram2dcnn_params
        feature_extractor = feature_extractors.CNNSpectgram(
            in_channels=params["in_channels"],
            base_filters=params["base_filters"],
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            sigmoid=params["sigmoid"],
            output_size=params["output_size"] // params["downsample_rate"],
        )
        decoder = decoders.Unet1DDecoder(
            n_channels=feature_extractor.height,
            n_classes=params["n_classes"],
            duration=params["seq_len"] // params["downsample_rate"],
            bilinear=params["bilinear"],
            se=params["se"],
            res=params["res"],
            scale_factor=params["scale_factor"],
            dropout=params["dropout"],
        )
        if params["use_spec_augment"]:
            assert params["spec_augment_params"] is not None
            spec_augment = augmentations.made_spec_augment_func(
                **params["spec_augment_params"]
            )
        else:
            spec_augment = None
        model = Spectrogram2DCNN(
            feature_extractor,
            decoder,
            encoder_name=params["encoder_name"],
            in_channels=feature_extractor.out_chans,
            encoder_weights=params["encoder_weights"],
            use_sample_weights=params["use_sample_weights"],
            spec_augment=spec_augment,
            use_aux_head=params.get("use_aux_head", False),
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


def _test_run_model4():
    print("Test3")
    # device = torch.device("cuda")
    device = torch.device("cpu")
    model = SleepTransformer(
        model_dim=10,
        dropout_rate=0.2,
        num_encoder_layers=3,
        num_lstm_layers=3,
        embed_dim=10,
        num_heads=2,
        seq_model_dim=10,
        seq_len=3000,
        device=device,
    )
    model = model.to(device).train()

    max_chunk_size = 10
    seq_len = 3000
    bs = 8 * 2
    x = torch.randn(bs, seq_len, max_chunk_size).to(device)
    print(x.shape)
    p = model(x, training=True)
    print("pred: ", p.shape)


def _test_run_model5():
    from src import utils

    print("Test5")
    # device = torch.device("cuda")
    device = torch.device("cpu")
    seq_len_ = 32 * 16 * 10
    downsample_rate = 2
    params: dict[str, Any] = dict(
        downsample_rate=downsample_rate,
        # -- CNNSpectgram
        in_channels=4,
        base_filters=64 * 4,
        kernel_size=[32, 16, downsample_rate],
        stride=downsample_rate,
        sigmoid=True,
        output_size=seq_len_,
        # -- Unet1DDecoder
        n_classes=3,
        duration=seq_len_,
        bilinear=False,
        se=False,
        res=False,
        scale_factor=2,
        dropout=0.2,
        # -- Spectrogram2DCNN
        # encoder_name="resnet18",
        # encoder_name="maxvit_tiny_rw_256",  # これ動かすにはカスタム必要
        # encoder_name="maxvit_tiny_rw_256",  # これ動かすにはカスタム必要
        encoder_name="maxvit_rmlp_nano_rw_256.sw_in1k",  # これ動かすにはカスタム必要
        # encoder_name="mit_b0",
        encoder_weights="imagenet",
        # encoder_weights=None,
    )
    feature_extractor = feature_extractors.CNNSpectgram(
        in_channels=params["in_channels"],
        base_filters=params["base_filters"],
        kernel_size=params["kernel_size"],
        stride=params["stride"],
        sigmoid=params["sigmoid"],
        output_size=params["output_size"] // params["downsample_rate"],
    )
    decoder = decoders.Unet1DDecoder(
        n_channels=feature_extractor.height,
        n_classes=params["n_classes"],
        duration=params["duration"] // params["downsample_rate"],
        bilinear=params["bilinear"],
        se=params["se"],
        res=params["res"],
        scale_factor=params["scale_factor"],
        dropout=params["dropout"],
    )
    model = Spectrogram2DCNN(
        feature_extractor,
        decoder,
        encoder_name=params["encoder_name"],
        in_channels=feature_extractor.out_chans,
        encoder_weights=params["encoder_weights"],
        use_sample_weights=False,
        use_aux_head=False,
    )
    model = model.to(device).train()

    import pathlib
    from pathlib import Path

    from src import dataset

    class Config:
        seed: int = 42
        num_workers: int = 0
        # Used in build_dataloader
        window_size: int = 10
        root_dir: Path = Path(".")
        input_dir: Path = root_dir / "input"
        data_dir: Path = input_dir / "child-mind-institute-detect-sleep-states"
        train_series_path: str | Path = (
            input_dir / "for_train" / "train_series_fold.parquet"
        )
        train_events_path: str | Path = data_dir / "train_events.csv"
        test_series_path: str | Path = data_dir / "test_series.parquet"

        out_size: int = 3
        series_dir: Path = Path("./output/series")
        target_series_uni_ids_path: Path = series_dir / "target_series_uni_ids.pkl"

        data_dir: pathlib.Path = pathlib.Path(
            "./input/child-mind-institute-detect-sleep-states"
        )
        processed_dir: pathlib.Path = pathlib.Path("./input/processed")
        seed: int

        train_series: list[str] = [
            "3df0da2e5966",
            "05e1944c3818",
            "bfe41e96d12f",
            "062dbd4c95e6",
            "1319a1935f48",
            "67f5fc60e494",
            "d2d6b9af0553",
            "aa81faa78747",
            "4a31811f3558",
            "e2a849d283c0",
            "361366da569e",
            "2f7504d0f426",
            "e1f5abb82285",
            "e0686434d029",
            "6bf95a3cf91c",
            "a596ad0b82aa",
            "8becc76ea607",
            "12d01911d509",
            "a167532acca2",
        ]
        valid_series: list[str] = [
            "e0d7b0dcf9f3",
            "519ae2d858b0",
            "280e08693c6d",
            "25e2b3dd9c3b",
            "9ee455e4770d",
            "0402a003dae9",
            "78569a801a38",
            "b84960841a75",
            "1955d568d987",
            "599ca4ed791b",
            "971207c6a525",
            "def21f50dd3c",
            "8fb18e36697d",
            "51b23d177971",
            "c7b1283bb7eb",
            "2654a87be968",
            "af91d9a50547",
            "a4e48102f402",
        ]

        # seq_len: int = 24 * 60 * 4
        seq_len: int = seq_len_
        """系列長の長さ"""
        features: list[str] = ["anglez", "enmo", "hour_sin", "hour_cos"]

        batch_size: int = 8

        upsample_rate: float = 1.0
        """default: 1.0"""
        downsample_rate: int = 2
        """default: 2"""

        bg_sampling_rate: float = 0.5
        """negative labelのサンプリング率. default: 0.5"""
        offset: int = 10
        """gaussian labelのoffset. default: 10"""
        sigma: int = 10
        """gaussian labelのsigma. default: 10"""
        sample_per_epoch: int | None = None

    dl = dataset.init_dataloader("train", Config)
    batch = next(iter(dl))
    x = batch["feature"].to(device)
    y = batch["label"].to(device)
    print(x.shape, y.shape)

    # x = torch.randn(bs, num_features, seq_len).to(device)
    # print(x.shape)
    with utils.trace("model forward"):
        p = model(x, y)
    print(type(p))
    print(p["logits"].shape)
    print(p["loss"].shape)
    print(p["loss"])


if __name__ == "__main__":
    # _test_run_model()
    # _test_run_model2()
    # _test_run_model3()
    # _test_run_model4()
    _test_run_model5()
