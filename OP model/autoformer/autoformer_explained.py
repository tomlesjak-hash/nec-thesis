"""
autoformer_explained.py

this file is a cleaned and commented reconstruction of the autoformer code shown in the
provided screenshots. i translated the original chinese comments into english and added
substantially more explanation so the flow of tensors and the intent of each block are clear.

important note:
- this is meant to preserve the logic shown in the screenshots as closely as possible.
- one part of the provided decoder layer appears suspicious: it computes an extra conv/mlp
  branch but still returns `x` instead of the transformed `y`. i preserved that behavior,
  because the request was to convert the screenshots into code rather than silently rewrite
  the architecture.
- if you want, that decoder branch can be audited and fixed later.

expected input convention in this file:
- batch-first tensors with shape (B, L, F)
  B = batch size
  L = sequence length / lookback window
  F = number of input features

this implementation is a simplified autoformer-like forecasting model with:
- token/value embedding
- optional positional embedding
- optional time-feature embedding
- fft-based auto-correlation attention
- encoder / decoder stacks

it is useful for long-horizon time-series forecasting when the data contains recurring lag
structure or periodicity.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# attention part
# -----------------------------------------------------------------------------
class AutoCorrelation(nn.Module):
    """
    auto-correlation mechanism.

    instead of vanilla dot-product attention, this block looks for repeating lag patterns.
    it does this by:
    1. moving queries and keys into the frequency domain with FFT,
    2. computing correlation there,
    3. transforming back,
    4. selecting the top-k delays (lags),
    5. aggregating shifted values according to those delays.

    why this exists:
    standard attention compares every time step against every other time step.
    this mechanism is built to emphasize periodic / recurring temporal dependencies more
    directly, which is the central idea behind autoformer-style forecasting.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 1,
        scale=None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values: Tensor, corr: Tensor) -> Tensor:
        """
        speed-optimized time-delay aggregation for the training phase.

        inputs:
        - values: (B, H, D, L)
            B = batch size
            H = number of heads
            D = value dimension per head
            L = sequence length
        - corr: (B, H, E, L)
            correlation scores over different delays

        training-time design choice:
        the top delays are selected using a batch-level average signal rather than a fully
        per-sample selection. this is cheaper and acts somewhat like a batch-normalization-
        style simplification.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # choose how many delays to keep.
        # original comment translated: "hyperparameter / scaling factor"
        top_k = int(self.factor * math.log(length))

        # average over heads and embedding dimensions first:
        # corr: (B, H, E, L) -> (B, L)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)

        # avoid asking for more delays than exist.
        if top_k > mean_value.size(-1):
            top_k = mean_value.size(-1)

        # find the most important delays using the batch-average score across B.
        # index has shape (top_k,)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        # collect the per-batch weights for those selected delays.
        # weights shape: (B, top_k)
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        # normalize selected delay scores into weights.
        tmp_corr = torch.softmax(weights, dim=-1)

        # initialize output buffer.
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)
        tmp_values = values

        # for each selected delay:
        #   1. roll the values along the time axis,
        #   2. weight that shifted pattern,
        #   3. add it into the aggregated output.
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )

        return delays_agg

    def time_delay_agg_inference(self, values: Tensor, corr: Tensor) -> Tensor:
        """
        speed-optimized time-delay aggregation for inference.

        unlike the training version, this one picks top delays per sample rather than from a
        global batch-average delay index.

        inputs:
        - values: (B, H, D, L)
        - corr:   (B, H, E, L)
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # base time index used for gather.
        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )

        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)

        if top_k > mean_value.size(-1):
            top_k = mean_value.size(-1)

        # per-sample top-k delays and their scores.
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]

        tmp_corr = torch.softmax(weights, dim=-1)

        # repeat values so delayed gathers beyond the end still work naturally.
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values, dtype=torch.float, device=values.device)

        for i in range(top_k):
            tmp_delay = init_index + (
                delay[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )

        return delays_agg

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        queries: (B, L, H, E)
        keys:    (B, S, H, E)
        values:  (B, S, H, D)

        returns:
        - V: aggregated values, shape (B, L, H, D)
        - optionally the correlation tensor if output_attention=True
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # if the query length is longer than the key/value length, pad key/value side.
        if L > S:
            zeros = torch.zeros_like(
                queries[:, :(L - S), :], dtype=torch.float, device=queries.device
            )
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependency search via FFT.
        # permute to put sequence length on the last axis before FFT.
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        # correlation in frequency domain uses complex conjugate.
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # aggregate delayed value patterns.
        # values.permute(0, 2, 3, 1) => (B, H, D, L)
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    """
    wrapper that projects model-space representations into multi-head query/key/value tensors,
    applies the auto-correlation mechanism, then projects back to model space.
    """

    def __init__(
        self,
        correlation,
        hidden_size,
        n_head,
        d_keys=None,
        d_values=None,
    ):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_head)
        d_values = d_values or (hidden_size // n_head)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_head)
        self.value_projection = nn.Linear(hidden_size, d_values * n_head)
        self.out_projection = nn.Linear(d_values * n_head, hidden_size)
        self.n_head = n_head

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# -----------------------------------------------------------------------------
# special part for autoformer
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """
    special layer normalization used for the seasonal part.

    after standard layer norm, it subtracts the mean across the time dimension.
    this pushes the representation closer to a zero-mean seasonal component.
    """

    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


# -----------------------------------------------------------------------------
# embedding part
# -----------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """
    standard sinusoidal positional embedding.

    this gives the model information about relative location inside the input window.
    """

    def __init__(self, hidden_size: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()

        # compute positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape becomes (1, max_len, hidden_size) so it can broadcast across batch.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """
    projects raw input features into the model hidden dimension using a 1d convolution.

    intuition:
    each time step originally has F raw features. this block turns them into hidden_size
    latent features that the transformer-like stack can work with.
    """

    def __init__(self, c_in: int, hidden_size: int):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=hidden_size,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: Tensor) -> Tensor:
        # input x: (B, L, c_in)
        # permute for Conv1d -> (B, c_in, L)
        # output after transpose -> (B, L, hidden_size)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    """
    linear embedding for external time features.

    examples of time features:
    - hour of day
    - day of week
    - month of year
    - trading session flags
    - holiday or event indicators
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None:
            return None
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    combines:
    - token/value embedding from raw features,
    - optional positional embedding,
    - optional time-feature embedding,
    - dropout.
    """

    def __init__(
        self,
        c_in: int,
        hidden_size: int,
        time_fea_size: int,
        pos_embedding: bool = True,
        dropout: float = 0.1,
    ):
        super(DataEmbedding, self).__init__()

        # translate original chinese note:
        # "transform dimension; c_in is the number of feature dimensions of the input data"
        self.value_embedding = TokenEmbedding(c_in=c_in, hidden_size=hidden_size)

        if pos_embedding:
            self.position_embedding = PositionalEmbedding(hidden_size=hidden_size)
        else:
            self.position_embedding = None

        if time_fea_size > 0:
            self.temporal_embedding = TimeFeatureEmbedding(
                input_size=time_fea_size, hidden_size=hidden_size
            )
        else:
            self.temporal_embedding = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Optional[Tensor] = None) -> Tensor:
        # base convolutional / token embedding.
        x = self.value_embedding(x)

        # add relative-within-window position information.
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)

        # add absolute time features if available.
        if self.temporal_embedding is not None:
            x_mark_embedding = self.temporal_embedding(x_mark)
            if x_mark_embedding is not None:
                x = x + x_mark_embedding

        return self.dropout(x)


# -----------------------------------------------------------------------------
# coder part
# -----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    autoformer encoder layer.

    sequence:
    1. auto-correlation attention
    2. residual add
    3. point-wise conv feed-forward branch
    4. additional mlp branch (as in the screenshot code)

    note:
    the original screenshots labelled this as a progressive decomposition architecture, but
    the explicit decomposition block is not present in these screenshots. this file only
    reconstructs the code that was shown.
    """

    def __init__(
        self,
        attention,
        hidden_size,
        conv_hidden_size=None,
        dropout=0.1,
        activation="relu",
    ):
        super(EncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # extra MLP branch from the screenshot.
        self.mlp_hidden_size = hidden_size * 2
        self.fc1 = nn.Linear(hidden_size, self.mlp_hidden_size)
        self.fc2 = nn.Linear(self.mlp_hidden_size, hidden_size)

    def forward(
        self, x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # extra MLP branch shown in the screenshots.
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.fc2(y)

        return y, attn


class Encoder(nn.Module):
    """
    stack of encoder layers.
    optionally supports interleaved conv layers, though none are constructed in this file.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(
        self, x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        attns = []

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)

            x, attn = self.attn_layers[-1](x)
            if attn is not None:
                attns.append(attn)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                if attn is not None:
                    attns.append(attn)

        # note: in the screenshot code self.norm is stored but never applied here.
        return x, attns


class DecoderLayer(nn.Module):
    """
    autoformer decoder layer.

    sequence:
    1. self auto-correlation attention on decoder input
    2. cross auto-correlation attention with encoder output
    3. point-wise conv branch
    4. mlp branch

    important note:
    the screenshot code returns `x` instead of the processed `y`. that means the conv/mlp
    branch is effectively ignored in the output. this is preserved exactly here.
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        hidden_size,
        c_out,
        conv_hidden_size=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

        # original chinese note translated:
        # "project trend with a 1d convolution down to output dimension"
        self.projection = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

        self.activation = F.relu if activation == "relu" else F.gelu
        self.mlp_hidden_size = hidden_size * 2
        self.fc1 = nn.Linear(hidden_size, self.mlp_hidden_size)
        self.fc2 = nn.Linear(self.mlp_hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        x_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # extra MLP branch shown in the screenshots.
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.fc2(y)

        # preserved exactly from the screenshot code.
        return x


class Decoder(nn.Module):
    """
    stack of decoder layers plus optional final projection.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        x_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # original chinese note translated:
        # "here no trend argument is additionally passed in"
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.projection is not None:
            x = self.projection(x)
        return x


# -----------------------------------------------------------------------------
# full model
# -----------------------------------------------------------------------------
class Autoformer(nn.Module):
    """
    simplified autoformer forecasting model.

    parameters:
    - enc_in: number of encoder input features
    - dec_in: number of decoder input features
    - c_out: number of output features after final projection
    - out_len: forecast horizon
    - time_fea_size: dimension of optional calendar / time covariates
    - hidden_size: model width
    - factor: controls how many dominant delays are selected
    - n_head: number of heads in the auto-correlation layer
    - conv_hidden_size: inner feed-forward width for 1x1 conv branches
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        out_len: int,
        time_fea_size: int = 0,
        hidden_size: int = 128,
        dropout: float = 0.05,
        factor: int = 3,
        n_head: int = 4,
        conv_hidden_size: int = 32,
        activation: str = "gelu",
        encoder_layers: int = 2,
        decoder_layers: int = 1,
    ):
        super(Autoformer, self).__init__()
        self.output_attention = False
        self.pred_len = out_len

        # embeddings for encoder and decoder streams.
        self.enc_embedding = DataEmbedding(
            enc_in,
            hidden_size=hidden_size,
            time_fea_size=time_fea_size,
            pos_embedding=False,
            dropout=dropout,
        )

        self.dec_embedding = DataEmbedding(
            dec_in,
            hidden_size=hidden_size,
            time_fea_size=time_fea_size,
            pos_embedding=False,
            dropout=dropout,
        )

        # encoder stack
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(encoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
        )

        # decoder stack
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    c_out=c_out,
                    conv_hidden_size=conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(decoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, c_out, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, F)

        output:
        - forecast over the last pred_len steps
        - if c_out == 1, returns shape roughly (B, pred_len)

        decoder construction logic used in the screenshot:
        - take the most recent part of the input sequence,
        - append zeros for the future horizon,
        - let the decoder fill in the unknown future steps.
        """
        # no external time-feature tensors are provided in this simplified version.
        x_mark_enc = x_mark_dec = None

        x_enc = x
        zeros = torch.zeros(
            size=(len(x), self.pred_len, x.shape[-1]),
            device=x.device,
        )

        # original chinese note translated:
        # "take the latter half / recent segment and append future placeholders"
        x_dec = torch.cat(
            [x[:, -(x.shape[1] - self.pred_len) :, :], zeros],
            dim=1,
        )

        # encoder pass
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)

        # decoder pass
        x_dec = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(x_dec, enc_out)

        # original chinese note translated:
        # "take the third dimension / output channel as the prediction"
        forecast = dec_out[:, -self.pred_len :, 0]
        return forecast.squeeze(-1)


# -----------------------------------------------------------------------------
# big-picture note for quant use
# -----------------------------------------------------------------------------
# why this can be used in a quant model:
#
# autoformer-style models are forecasting architectures for structured time-series data.
# in quant, that means they are not magic alpha generators; they are tools for predicting
# something about the future path of a series.
#
# examples of quant targets where this kind of architecture can make sense:
# - realized volatility
# - intraday volume curves
# - order-flow imbalance
# - spread dynamics
# - inventory pressure
# - factor values that evolve with recurring temporal structure
# - macro / rates / cross-asset signals with seasonality or persistent lag effects
#
# why auto-correlation is attractive in finance:
# - some financial series are not well described by only local lags.
# - many series have repeating structures: market open / close behavior, day-of-week effects,
#   option expiry patterns, intraday liquidity cycles, volatility clustering, etc.
# - FFT-based correlation can be a compact way to search for useful delays across longer
#   horizons than a plain short-memory model.
#
# but the hard truth:
# - this model is only useful if the target actually has forecastable structure.
# - for liquid-asset raw returns, the signal-to-noise ratio is often terrible.
# - if you do not control for leakage, walk-forward validation, transaction costs, and regime
#   changes, a fancy architecture will not save you.
# - you still need to benchmark it against simpler models like linear regression, tree models,
#   TCN/LSTM baselines, and naive lag forecasts.
#
# so the correct way to think about it is:
# this is a time-series forecasting engine that may help extract recurring temporal structure.
# whether that becomes a tradable quant edge depends on target design, feature quality,
# validation discipline, and execution realism.
