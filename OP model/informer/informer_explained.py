"""
informer_explained.py

this file is a reconstructed and heavily commented version of the informer-style code
shown in the screenshots.

i translated the chinese comments into english and added much more explanation so you
can see what each block is doing, how tensor shapes move, and why this architecture is
used in quant / time-series forecasting work.

important note:
- this reconstruction follows the screenshot logic as closely as possible.
- some details in the screenshots rely on externally provided qk index tensors
  (`enc_qk_idxes` and `dec_qk_idxes`), which are not standard in the original
  informer paper code. those tensors are preserved here as explicit inputs.
- a few implementation details were cleaned slightly so the code is readable and
  runnable, but the architecture and flow stay aligned with what appears in the images.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# convolutional distillation block used inside the encoder
# -----------------------------------------------------------------------------
class ConvLayer(nn.Module):
    """
    conv layer used inside the encoder.

    informer uses these blocks between encoder layers when `distil=True`.
    the idea is to reduce sequence length while keeping feature dimension fixed,
    which makes deeper attention stacks cheaper.
    """

    def __init__(self, c_in: int) -> None:
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2

        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )

        # batchnorm1d here normalizes along the channel axis for conv1d output.
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, C)
        returns: (B, L_reduced, C)
        """
        # move features to the channel dimension for Conv1d
        x = self.downConv(x.permute(0, 2, 1))   # (B, C, L)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)                     # sequence length gets reduced
        x = x.transpose(1, 2)                   # back to (B, L, C)
        return x


# -----------------------------------------------------------------------------
# probabilistic attention used by informer
# -----------------------------------------------------------------------------
class ProbAttention(nn.Module):
    """
    informer probabilistic attention.

    standard full attention computes every query-key interaction.
    prob attention tries to cut cost by focusing only on the most informative queries.

    this version differs slightly from the reference implementation because the
    screenshot code uses externally provided `qk_idx` indices to choose sampled keys.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self, Q: Tensor, K: Tensor, n_top: int, qk_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        compute sparse query-key scores.

        parameters
        ----------
        Q : Tensor
            shape (B, H, L_Q, D)
        K : Tensor
            shape (B, H, L_K, D)
        n_top : int
            number of top queries to keep based on sparsity score.
        qk_idx : Tensor
            external sampled key indices for each query.
            expected shape is roughly (L_Q, sample_k).

        returns
        -------
        Q_K : Tensor
            attention scores for only the selected top queries,
            shape (B, H, n_top, L_K)
        M_top : Tensor
            indices of the selected top queries, shape (B, H, n_top)
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # expand K so every query position can gather its sampled keys.
        # K_expand: (B, H, L_Q, L_K, E)
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # qk_idx is expanded to batch/head dimensions, then used to gather
        # sampled keys for each query.
        qk_idx = qk_idx.to(K.device)
        qk_idx_expanded = qk_idx.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        qk_idx_expanded = qk_idx_expanded.unsqueeze(-1)

        # sampled keys per query: (B, H, L_Q, sample_k, E)
        K_sample = torch.gather(
            K_expand,
            dim=3,
            index=qk_idx_expanded.expand(-1, -1, -1, -1, E),
        )

        # score sampled keys against each query.
        # Q.unsqueeze(-2): (B, H, L_Q, 1, E)
        # result: (B, H, L_Q, sample_k)
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)
        ).squeeze(-2)

        # sparsity measure: max score minus mean score.
        # informative queries should stand out against the average.
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # pick the top-n_top most informative queries.
        M_top = M.topk(n_top, sorted=False)[1]

        # gather only those top queries from Q.
        Q_reduce = torch.gather(
            Q,
            dim=2,
            index=M_top.unsqueeze(-1).expand(-1, -1, -1, E),
        )

        # now compute these selected queries against all keys.
        # output: (B, H, n_top, L_K)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V: Tensor, L_Q: int) -> Tensor:
        """
        build the initial context before selected queries are updated.

        if not masked:
        use the mean of V for every query position.

        if masked:
        use cumulative sums, which is appropriate for causal self-attention.
        """
        B, H, L_V, D = V.shape

        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V  # only valid for self-attention in the masked case
            context = V.cumsum(dim=-2)

        return context

    def _update_context(
        self,
        context_in: Tensor,
        V: Tensor,
        scores: Tensor,
        index: Tensor,
        L_Q: int,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        update context only at the selected top-query positions.

        parameters
        ----------
        context_in : Tensor
            current context, shape (B, H, L_Q, D)
        V : Tensor
            values, shape (B, H, L_V, D)
        scores : Tensor
            scores for selected queries, shape (B, H, n_top, L_K)
        index : Tensor
            selected query indices, shape (B, H, n_top)
        L_Q : int
            number of queries
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            device = V.device

            # upper-triangular causal mask over query-key pairs.
            base_mask = torch.ones(
                L_Q, scores.shape[-1], dtype=torch.bool, device=device
            ).triu(1)

            # expand to all batch/head dimensions.
            mask_ex = base_mask[None, None, :, :].repeat(B, H, 1, 1)

            # keep only the mask rows for selected top queries.
            indicator = torch.gather(
                mask_ex,
                dim=2,
                index=index.unsqueeze(-1).expand(-1, -1, -1, mask_ex.shape[-1]),
            ).to(device)

            mask = indicator.view(scores.shape).to(device)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # update context at selected query positions with weighted values.
        context_in.scatter_(
            dim=2,
            index=index.unsqueeze(-1).expand(-1, -1, -1, D),
            src=torch.matmul(attn, V).type_as(context_in),
        )

        if self.output_attention:
            # build a full attention tensor for output compatibility.
            attns = (torch.ones([B, H, L_V, L_V], device=attn.device) / L_V).type_as(attn)
            attns.scatter_(
                dim=2,
                index=index.unsqueeze(-1).expand(-1, -1, -1, L_V),
                src=attn,
            )
            return context_in, attns
        else:
            return context_in, None

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        qk_idx: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        queries: (B, L_Q, H, D)
        keys:    (B, L_K, H, D)
        values:  (B, L_K, H, D)

        returns
        -------
        context : (B, L_Q, H, D)
        attn    : optional full attention tensor
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # switch to (B, H, L, D) because the attention implementation works in that layout.
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # number of top queries to keep: c * ln(L_Q)
        u = self.factor * torch.ceil(torch.log(torch.tensor(L_Q, dtype=torch.float32)))
        u = torch.minimum(u, torch.tensor(L_Q, dtype=torch.float32))
        u = int(u.to(torch.int64).item())
        u = max(u, 1)

        scores_top, index = self._prob_QK(queries, keys, n_top=u, qk_idx=qk_idx)

        # scaled dot-product factor
        if self.scale is None:
            scale = 1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32, device=queries.device))
        else:
            scale = torch.tensor(self.scale, dtype=torch.float32, device=queries.device)

        scores_top = scores_top * scale

        # initialize then update context using selected top queries.
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        # back to (B, L_Q, H, D)
        return context.transpose(2, 1).contiguous(), attn


# -----------------------------------------------------------------------------
# multi-head wrapper around inner attention
# -----------------------------------------------------------------------------
class AttentionLayer(nn.Module):
    """
    generic multi-head attention wrapper.

    this takes a low-level attention module (here, ProbAttention), projects inputs into
    heads, applies attention, then projects back to the model dimension.
    """

    def __init__(
        self,
        attention: nn.Module,
        hidden_size: int,
        n_head: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hidden_size // n_head)
        d_values = d_values or (hidden_size // n_head)

        self.inner_attention = attention
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
        qk_idx: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        queries: (B, L, hidden_size)
        keys:    (B, S, hidden_size)
        values:  (B, S, hidden_size)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, qk_idx.to(keys.device))
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# -----------------------------------------------------------------------------
# embedding blocks
# -----------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()

        # standard sinusoidal positional encoding
        pe = torch.zeros(max_len, hidden_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape becomes (1, max_len, hidden_size) so it can broadcast across batch.
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """
    local convolutional projection from raw feature space into model hidden space.
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
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: Tensor) -> Tensor:
        # (B, L, C) -> (B, C, L) -> conv -> (B, hidden_size, L) -> (B, L, hidden_size)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    """
    optional linear embedding for externally supplied time features.
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
    - value/token embedding
    - optional positional embedding
    - optional time-feature embedding
    - dropout
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

        self.value_embedding = TokenEmbedding(c_in=c_in, hidden_size=hidden_size)

        if pos_embedding:
            self.position_embedding = PositionalEmbedding(hidden_size=hidden_size)
        else:
            self.position_embedding = None

        if time_fea_size > 0:
            self.temporal_embedding = TimeFeatureEmbedding(
                input_size=time_fea_size,
                hidden_size=hidden_size,
            )
        else:
            self.temporal_embedding = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Optional[Tensor] = None) -> Tensor:
        x = self.value_embedding(x)

        # relative position information within the window
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)

        # optional absolute / calendar time features
        if self.temporal_embedding is not None:
            x_mark_embedding = self.temporal_embedding(x_mark)
            if x_mark_embedding is not None:
                x = x + x_mark_embedding

        return self.dropout(x)


# -----------------------------------------------------------------------------
# encoder blocks
# -----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    transformer-style encoder block:
    attention -> residual -> norm -> feedforward -> residual -> norm
    """

    def __init__(
        self,
        attention: nn.Module,
        hidden_size: int,
        conv_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(EncoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: Tensor,
        enc_qk_idx: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        new_x, attn = self.attention(x, x, x, enc_qk_idx)
        x = x + self.dropout(new_x)

        # normalize before feedforward block
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    informer encoder.

    when `conv_layers` is provided, sequence length is reduced between attention blocks.
    `enc_qk_idxes` stores external sampled key indices for each encoder attention layer.
    """

    def __init__(
        self,
        attn_layers: List[nn.Module],
        enc_qk_idxes: Tuple[Tensor],
        conv_layers: Optional[List[nn.Module]] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.enc_qk_idxes = enc_qk_idxes
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        attns: List[Tensor] = []

        if self.conv_layers is not None:
            # for all but the last encoder layer: attention + conv distillation
            for i, conv_layer in enumerate(self.conv_layers):
                for j, attn_layer in enumerate(self.attn_layers):
                    if i == j:
                        enc_qk_idx = self.enc_qk_idxes[i]
                        x, attn = attn_layer(x, enc_qk_idx)
                        x = conv_layer(x)
                        if attn is not None:
                            attns.append(attn)

            # final attention layer has no distillation layer after it
            x, attn = self.attn_layers[-1](x, self.enc_qk_idxes[-1])
            if attn is not None:
                attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                enc_qk_idx = self.enc_qk_idxes[i]
                x, attn = attn_layer(x, enc_qk_idx)
                if attn is not None:
                    attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# -----------------------------------------------------------------------------
# decoder blocks
# -----------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    transformer-style decoder block with:
    - masked self-attention
    - cross-attention to encoder output
    - feedforward block
    """

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        hidden_size: int,
        conv_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(DecoderLayer, self).__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_hidden_size,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        dec_qk_idx: List[Tensor],
    ) -> Tensor:
        # masked self-attention over decoder input
        x = x + self.dropout(self.self_attention(x, x, x, dec_qk_idx[0])[0])
        x = self.norm1(x)

        # cross-attention: decoder queries attend to encoder output
        x = x + self.dropout(self.cross_attention(x, cross, cross, dec_qk_idx[1])[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(
        self,
        layers: List[nn.Module],
        dec_qk_idxes: Tuple[Tensor],
        norm_layer: Optional[nn.Module] = None,
        projection: Optional[nn.Module] = None,
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.dec_qk_idxes = dec_qk_idxes
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: Tensor, cross: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, cross, self.dec_qk_idxes[i])

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x


# -----------------------------------------------------------------------------
# full informer model
# -----------------------------------------------------------------------------
class Informer(nn.Module):
    """
    full encoder-decoder informer.

    expected external inputs:
    - enc_qk_idxes: sampled key indices per encoder layer
    - dec_qk_idxes: sampled key indices per decoder layer
      each decoder entry should contain two tensors:
      [self_attention_qk_idx, cross_attention_qk_idx]
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        out_len: int,
        enc_qk_idxes: List[Tensor],
        dec_qk_idxes: List[List[Tensor]],
        time_fea_size: int = 0,
        factor: int = 5,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        conv_hidden_size: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = True,
    ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        # embeddings
        self.enc_embedding = DataEmbedding(
            enc_in,
            hidden_size,
            time_fea_size,
            pos_embedding=True,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            dec_in,
            hidden_size,
            time_fea_size,
            pos_embedding=True,
            dropout=dropout,
        )

        # encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            enc_qk_idxes,
            [ConvLayer(hidden_size) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

        # decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            dec_qk_idxes=dec_qk_idxes,
            norm_layer=torch.nn.LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, c_out, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, F)

        the screenshot code constructs decoder input from the second half of x,
        then appends zero rows for the future steps to be predicted.
        """
        x_mark_enc = x_mark_dec = None

        # encoder input: full sequence
        # decoder input: second half of sequence + zero placeholders for future horizon
        x_enc, x_dec = x, x[:, -x.shape[1] // 2 :, :]
        x_dec = torch.cat(
            [
                x_dec,
                torch.zeros(
                    [x_dec.shape[0], self.pred_len - 1, x_dec.shape[-1]],
                    device=x.device,
                ),
            ],
            dim=1,
        )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)

        # return only the final prediction window
        return dec_out[:, -self.pred_len :, :]


# -----------------------------------------------------------------------------
# encoder-only informer variant
# -----------------------------------------------------------------------------
class InformerEncOnly(nn.Module):
    """
    encoder-only informer.

    instead of an autoregressive decoder, this version flattens the final encoder output
    and feeds it into a small MLP head.

    this is often more convenient for classification, regression, or one-shot prediction
    tasks where you do not need a full sequence decoder.
    """

    def __init__(
        self,
        enc_in: int,
        c_out: int,
        enc_qk_idxes: List[Tensor],
        time_fea_size: int = 0,
        factor: int = 5,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        conv_hidden_size: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = True,
    ):
        super(InformerEncOnly, self).__init__()
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(
            enc_in,
            hidden_size,
            time_fea_size,
            pos_embedding=True,
            dropout=dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            enc_qk_idxes,
            [ConvLayer(hidden_size) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

        # final flattened encoder representation -> MLP head
        self.decoder = nn.Sequential(
            nn.Linear(enc_qk_idxes[-1].shape[0] * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, c_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_mark_enc = None

        x_enc = x
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)

        # flatten encoder output across time and hidden dimensions
        dec_out = enc_out.flatten(start_dim=1)
        dec_out = self.decoder(dec_out)

        return dec_out


# -----------------------------------------------------------------------------
# big picture: why this is used in a quant model
# -----------------------------------------------------------------------------
# informer is a long-sequence forecasting architecture.
# its core sales pitch is that full transformer attention is expensive for long windows,
# so informer uses probabilistic sparse attention and optional sequence distillation to
# reduce cost.
#
# in quant, this kind of model is used when you want to map a long history of features
# into a future prediction without relying only on simple lags or a small recurrent state.
#
# typical inputs:
# - returns over many bars / days
# - realized volatility history
# - volume and turnover patterns
# - spreads and order-book features
# - macro features or cross-asset signals
# - calendar / time features
#
# typical outputs:
# - next-k-step return forecasts
# - volatility forecasts
# - multi-step path forecasts
# - classification labels (up/down, regime, event response)
# - one-shot alpha scores in the encoder-only version
#
# why informer can be attractive in quant:
# - can handle longer lookback windows than many simple RNN setups
# - attention can model long-range dependencies more flexibly than a plain MLP
# - encoder distillation reduces sequence length and therefore computation
# - encoder-only version is convenient for factor-style prediction heads
#
# what it does NOT solve by itself:
# - no leakage prevention
# - no transaction cost awareness
# - no guarantee of economic meaning
# - no portfolio construction logic
# - no execution or slippage modeling
#
# bluntly: informer is just a sequence model.
# whether it is useful in a real quant pipeline depends much more on:
# - target design
# - feature engineering
# - data cleaning
# - strict walk-forward validation
# - cost-aware backtesting
# - robustness across regimes
#
# if those parts are weak, using informer instead of a GRU, CNN, or simpler baseline
# will not save the project.
