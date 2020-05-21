# -*- coding: utf-8 -*-

import math
from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from entmax import Entmax15, EntmaxBisect

from joeynmt.better_sparsemax import BetterSparsemax


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1,
                 attn_func: str = "softmax", attn_alpha: float = 1.5):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)

        Entmax = partial(EntmaxBisect, alpha=attn_alpha, n_iter=30)
        attn_funcs = {"softmax": nn.Softmax,
                      "sparsemax": BetterSparsemax,
                      "entmax15": Entmax15,
                      "entmax": Entmax}
        assert attn_func in attn_funcs, "Unknown attention function"
        self.transform = attn_funcs[attn_func](dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.transform(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)

        return output, attention


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self, size: int, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        position = position * div_term
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, dim)``
        """
        # Add position encodings
        return emb + self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self,
                 size: int,
                 ff_size: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attn_func: str = "softmax",
                 attn_alpha: float = 1.5):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(
            num_heads,
            size,
            dropout=dropout,
            attn_func=attn_func,
            attn_alpha=attn_alpha)
        self.feed_forward = PositionwiseFeedForward(size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h, self_attn = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o, self_attn


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self,
                 size: int,
                 ff_size: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 self_attn_func: str = "softmax",
                 self_attn_alpha: float = 1.5,
                 src_attn_func: str = "softmax",
                 src_attn_alpha: float = 1.5,
                 **kwargs):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        attn_mechanism = partial(
            MultiHeadedAttention,
            num_heads=num_heads,
            size=size,
            dropout=dropout)
        self.trg_trg_att = attn_mechanism(
            attn_func=self_attn_func, attn_alpha=self_attn_alpha)
        self.src_trg_att = attn_mechanism(
            attn_func=src_attn_func, attn_alpha=src_attn_alpha)

        # positionwise feedforward will always use default value
        self.feed_forward = PositionwiseFeedForward(size, ff_size, dropout)

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor,
                memory: Tensor,
                src_mask: Tensor = None,
                trg_mask: Tensor = None) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1, self_attn = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2, ctx_attn = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o, self_attn, ctx_attn


class MultiSourceTransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self,
                 size: int,
                 ff_size: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 self_attn_func: str = "softmax",
                 self_attn_alpha: float = 1.5,
                 src_attn_func: str = "softmax",
                 src_attn_alpha: float = 1.5,
                 gate_func: str = "softmax",
                 gate_alpha: float = 1.5,
                 merge: str = "serial",
                 **kwargs):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(MultiSourceTransformerDecoderLayer, self).__init__()
        assert isinstance(src_attn_func, str) or isinstance(src_attn_func, dict)
        assert isinstance(src_attn_alpha, float) or isinstance(src_attn_alpha, float)
        assert merge in ["serial", "parallel", "hierarchical", "real_hierarchical"]
        self.merge = merge
        self.size = size

        if merge == "hierarchical":
            Entmax = partial(EntmaxBisect, alpha=gate_alpha, n_iter=30)
            gate_funcs = {"softmax": nn.Softmax,
                          "sparsemax": BetterSparsemax,
                          "entmax15": Entmax15,
                          "entmax": Entmax}
            assert gate_func in gate_funcs, "Unknown gate function"
            transform = gate_funcs[gate_func](dim=-1)
            self.gate = nn.Sequential(nn.Linear(size * 2, 2), transform)
            self.hier_layer_norm = None
            self.hier_att = None
        elif merge == "real_hierarchical":
            self.gate = None
            self.hier_layer_norm = nn.LayerNorm(size, eps=1e-6)
            self.hier_att = MultiHeadedAttention(
                num_heads=num_heads,
                size=size,
                dropout=dropout,
                attn_func=gate_func,  # note
                attn_alpha=gate_alpha)
        else:
            self.gate = None
            self.hier_layer_norm = None
            self.hier_att = None

        attn_mechanism = partial(
            MultiHeadedAttention,
            num_heads=num_heads,
            size=size,
            dropout=dropout)
        self.trg_trg_att = attn_mechanism(
            attn_func=self_attn_func, attn_alpha=self_attn_alpha)

        self.enc_order = ["src", "inflection"]
        if isinstance(src_attn_func, str):
            src_attn_func = {enc: src_attn_func for enc in self.enc_order}
        if isinstance(src_attn_alpha, float):
            src_attn_alpha = {enc: src_attn_alpha for enc in self.enc_order}
        self.ctx_trg_atts = nn.ModuleDict()
        for enc in self.enc_order:
            self.ctx_trg_atts[enc] = attn_mechanism(
                attn_func=src_attn_func[enc], attn_alpha=src_attn_alpha[enc])

        # positionwise feedforward will always use default value
        self.feed_forward = PositionwiseFeedForward(size, ff_size, dropout)

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        # self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norms = nn.ModuleDict(
            {k: nn.LayerNorm(size, eps=1e-6) for k in self.enc_order})

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor,
                memories: dict,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                inflection_mask: Tensor = None) -> Tensor:
        """
        """
        enc_masks = {"src": src_mask, "inflection": inflection_mask}
        context_attns = dict()
        # a sublayer
        # 1) input_norm = layer_norm(input)
        # 2) attention between input_norm and (input_norm or memory),
        #    yielding attn_out and attn_dist
        # 3) output = dropout(attn_out) + input

        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        self_attn_out, self_attn = self.trg_trg_att(
            x_norm, x_norm, x_norm, mask=trg_mask
        )
        self_attn_out = self.dropout(self_attn_out) + x

        # serial-style encoder combination
        ctx_attn_in = self_attn_out
        ctx_attn_outs = []
        for enc_name in self.enc_order:
            # module and input lookup
            memory = memories[enc_name]
            mask = enc_masks[enc_name]
            layer_norm = self.dec_layer_norms[enc_name]
            attn = self.ctx_trg_atts[enc_name]

            # computation
            input_norm = layer_norm(ctx_attn_in)
            ctx_attn_out, attn_dist = attn(
                memory, memory, input_norm, mask=mask
            )
            context_attns[enc_name] = attn_dist

            if self.merge == "serial":
                ctx_attn_out = self.dropout(ctx_attn_out) + ctx_attn_in
                ctx_attn_in = ctx_attn_out
            else:
                ctx_attn_outs.append(ctx_attn_out)

        if self.merge in ["parallel", "hierarchical", "real_hierarchical"]:
            if self.merge != "real_hierarchical":
                stacked_ctx_attn = torch.stack(ctx_attn_outs, dim=-1)
                if self.merge == "parallel":
                    ctx_attn_out = stacked_ctx_attn.sum(dim=-1)
                elif self.merge == "hierarchical":
                    # a misnomer: this is similar to DeepSPIN's 2019 model,
                    # not to the hierarchical multi-encoder transformer
                    # proposed by Libovicky et al.
                    gate_weights = self.gate(
                        torch.cat(ctx_attn_outs, dim=-1)
                    ).unsqueeze(3)
                    ctx_attn_out = (stacked_ctx_attn @ gate_weights).squeeze(3)
            else:
                # real_hierarchical: this is Libovicky's hierarchical model
                '''
                catted_masks = torch.cat(
                    [enc_masks[enc_name] for enc_name in self.enc_order],
                    dim=-1
                )
                '''
                # print([a.size() for a in ctx_attn_outs])
                catted_ctx_attn = torch.cat(ctx_attn_outs, dim=1)

                # does the layer norm belong?
                hier_input_norm = self.hier_layer_norm(ctx_attn_in)
                # print(catted_masks.size(), catted_ctx_attn.size())
                ctx_attn_out, _ = self.hier_att(
                    catted_ctx_attn,
                    catted_ctx_attn,
                    hier_input_norm)
                
            ctx_attn_out = self.dropout(ctx_attn_out) + ctx_attn_in

        # final position-wise feed-forward layer
        o = self.feed_forward(ctx_attn_out)

        return o, self_attn, context_attns
