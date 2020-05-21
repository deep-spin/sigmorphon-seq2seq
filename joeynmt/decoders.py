# coding: utf-8

"""
Various decoders
"""
from collections import defaultdict
from functools import partial
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from entmax import entmax15, sparsemax, entmax_bisect

from joeynmt.attention import BahdanauAttention, LuongAttention, MultiAttention
from joeynmt.helpers import freeze_params, subsequent_mask
from joeynmt.transformer_layers import PositionalEncoding, \
    TransformerDecoderLayer, MultiSourceTransformerDecoderLayer


def log_sparsemax(*args, **kwargs):
    return torch.log(sparsemax(*args, **kwargs))


def log_entmax15(*args, **kwargs):
    return torch.log(entmax15(*args, **kwargs))


def log_entmax(*args, **kwargs):
    return torch.log(entmax_bisect(*args, **kwargs))


# pylint: disable=abstract-method
class Decoder(nn.Module):
    """
    Base decoder class
    """
    def __init__(
        self, hidden_size, vocab_size, emb_dropout: float = 0.,
        gen_func: str = "softmax", gen_alpha: float = 1.5,
        output_bias=False, **kwargs
    ):
        super(Decoder, self).__init__()

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # nn.ModuleDict to make MTL easier
        output_layer = nn.Linear(hidden_size, vocab_size, bias=output_bias)
        self.output_layers = nn.ModuleDict({"vocab": output_layer})
        self._vocab_size = vocab_size

        # the resulting gen_func is used only in a single line in search.py
        gen_funcs = {"softmax": F.log_softmax,
                     "sparsemax": log_sparsemax,
                     "entmax15": log_entmax15,
                     "entmax": partial(log_entmax, alpha=gen_alpha, n_iter=30)}
        assert gen_func in gen_funcs
        self.gen_func = gen_funcs[gen_func]

    @property
    def vocab_size(self):
        return self._vocab_size


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class RecurrentDecoder(Decoder):
    """A conditional RNN decoder with attention."""

    def __init__(self,
                 rnn_type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder_output_size: int = 0,
                 attention: str = "bahdanau",
                 num_layers: int = 1,
                 dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 init_hidden: str = "bridge",
                 input_feeding: bool = True,
                 freeze: bool = False,
                 attn_func: str = "softmax",
                 attn_alpha: float = 1.5,
                 **kwargs) -> None:
        """
        Create a recurrent decoder with attention.

        :param rnn_type: rnn type, valid options: "lstm", "gru"
        :param emb_size:
        :param hidden_size:
        :param attention: type of attention, valid options: "bahdanau", "luong"
        :param num_layers:
        :param hidden_dropout: applied to the input to the attentional layer.
        :param dropout: applied between RNN layers.
        :param emb_dropout: applied to the RNN input (word embeddings).
        :param init_hidden: If "bridge" (default), the decoder hidden states
            are initialized from a projection of the last encoder state,
            if "zeros" they are initialized with zeros,
            if "last" they are identical to the last encoder state
            (only if they have the same size)
        :param input_feeding: Use Luong's input feeding.
        :param freeze: Freeze the parameters of the decoder during training.
        :param kwargs: passed to generic Decoder constructor
        """

        super(RecurrentDecoder, self).__init__(hidden_size, **kwargs)

        self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        input_size = emb_size + hidden_size if input_feeding else emb_size

        # the decoder RNN
        self.rnn = rnn(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.)

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder_output_size, hidden_size, bias=True)

        assert attention in ["bahdanau", "luong"], \
            "Unknown attention mechanism: %s. Use 'bahdanau' or 'luong'."
        if attention == "bahdanau":
            attn_mechanism = partial(BahdanauAttention, query_size=hidden_size)
        else:
            attn_mechanism = LuongAttention
        self.attention = attn_mechanism(
            hidden_size=hidden_size,
            key_size=encoder_output_size,
            attn_func=attn_func,
            attn_alpha=attn_alpha)

        # to initialize from the final encoder state of last layer
        assert init_hidden in ["bridge", "zero", "last"]
        self.init_hidden_option = init_hidden
        if init_hidden == "bridge":
            self.bridge_layer = nn.Sequential(
                nn.Linear(encoder_output_size, hidden_size, bias=True),
                nn.Tanh()
            )
        else:
            self.bridge_layer = None
        if init_hidden == "last":
            out_size = encoder_output_size
            assert out_size in (hidden_size, 2 * hidden_size), \
                "Mismatched hidden sizes (encoder: {}, decoder: {})".format(
                    encoder_output_size, hidden_size
                )

        if freeze:
            freeze_params(self)

    @property
    def rnn_input_size(self):
        return self.rnn.input_size

    @property
    def num_layers(self):
        return self.rnn.num_layers

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    def _check_shapes_input_forward_step(self,
                                         prev_embed: Tensor,
                                         prev_att_vector: Tensor,
                                         encoder_output: Tensor,
                                         src_mask: Tensor,
                                         hidden: Tensor) -> None:
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        """
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size(
            [1, self.hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert src_mask.shape[0] == prev_embed.shape[0]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[2] == encoder_output.shape[1]
        if isinstance(hidden, tuple):  # for lstm
            hidden = hidden[0]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == self.hidden_size

    def _check_shapes_input_forward(self,
                                    trg_embed: Tensor,
                                    encoder_output: Tensor,
                                    encoder_hidden: Tensor,
                                    src_mask: Tensor,
                                    hidden: Tensor = None,
                                    prev_att_vector: Tensor = None) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param hidden:
        :param prev_att_vector:
        """
        assert len(encoder_output.shape) == 3
        assert len(encoder_hidden.shape) == 2
        assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[0] == encoder_output.shape[0]
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert trg_embed.shape[0] == encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):  # for lstm
                hidden = hidden[0]
            assert hidden.shape[1] == encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(self,
                      prev_embed: Tensor,
                      prev_att_vector: Tensor,  # context or att vector
                      encoder_output: Tensor,
                      src_mask: Tensor,
                      hidden: Tensor) -> (Tensor, Tensor, Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
            shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
            shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
            shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
            shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
            shape (num_layers, batch_size, hidden_size)
        :return:
            - att_vector: new attention vector (batch_size, 1, hidden_size),
            - hidden: new hidden state with shape (batch_size, 1, hidden_size),
            - att_probs: attention probabilities (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(prev_embed=prev_embed,
                                              prev_att_vector=prev_att_vector,
                                              encoder_output=encoder_output,
                                              src_mask=src_mask,
                                              hidden=hidden)

        if self.input_feeding:
            # concatenate the input with the previous attention vector
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        # compute context vector using attention mechanism
        # only use last layer for attention mechanism
        # key projections are pre-computed
        context, att_probs = self.attention(
            query=query, values=encoder_output, mask=src_mask)

        # return attention vector (Luong)
        # combine context with decoder hidden state before prediction
        att_vector_input = torch.cat([query, context], dim=2)
        # batch x 1 x 2*enc_size+hidden_size
        att_vector_input = self.hidden_dropout(att_vector_input)

        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs

    def _forward_pre_output(self,
                            trg_embed: Tensor,
                            encoder_output: Tensor,
                            encoder_hidden: Tensor,
                            src_mask: Tensor,
                            unroll_steps: int,
                            hidden: Tensor = None,
                            prev_att_vector: Tensor = None,
                            **kwargs):
        """
        Unroll the decoder one step at a time for `unroll_steps` steps.
        For every step, the `_forward_step` function is called internally.

        During training, the target inputs (`trg_embed') are already known for
        the full sequence, so the full unrol is done.
        In this case, `hidden` and `prev_att_vector` are None.

        For inference, this function is called with one step at a time since
        embedded targets are the predictions from the previous time step.
        In this case, `hidden` and `prev_att_vector` are fed from the output
        of the previous call of this function (from the 2nd step on).

        `src_mask` is needed to mask out the areas of the encoder states that
        should not receive any attention,
        which is everything after the first <eos>.

        The `encoder_output` are the hidden states from the encoder and are
        used as context for the attention.

        The `encoder_hidden` is the last encoder hidden state that is used to
        initialize the first hidden decoder state
        (when `self.init_hidden_option` is "bridge" or "last").

        :param trg_embed: emdedded target inputs,
        shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
        shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder,
        shape (batch_size x encoder.output_size)
        :param src_mask: mask for src states: 0s for padded areas,
        1s for the rest, shape (batch_size, 1, src_length)
        :param unroll_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
        if not given it's initialized as in `self.init_hidden`,
        shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
        if not given it's initialized with zeros,
        shape (batch_size, 1, hidden_size)
        :return:
        - outputs: shape (batch_size, unroll_steps, vocab_size),
        - hidden: last hidden state (num_layers, batch_size, hidden_size),
        - att_probs: attention probabilities
            with shape (batch_size, unroll_steps, src_length),
        - att_vectors: attentional vectors
            with shape (batch_size, unroll_steps, hidden_size)
        """
        # shape checks
        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            hidden=hidden,
            prev_att_vector=prev_att_vector)

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=encoder_output)

        # here we store intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unroll_steps, src_length

        # outputs: batch, unroll_steps, vocab_size
        return hidden, {"src_trg": att_probs}, att_vectors

    def forward(self,
                trg_embed: Tensor,
                encoder_output: Tensor,
                encoder_hidden: Tensor,
                src_mask: Tensor,
                unroll_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None,
                output_key: str = "vocab",
                language: Tensor = None,
                apply_mask: bool = True,
                **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):

        if isinstance(output_key, str):
            output_key = [output_key]
        assert all(k in self.output_layers for k in output_key)

        hidden, att_dict, att_vectors = self._forward_pre_output(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            unroll_steps=unroll_steps,
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            **kwargs)

        outputs = dict()
        for k in output_key:
            output_layer = self.output_layers[k]
            if isinstance(output_layer, nn.Linear):
                outputs[k] = output_layer(att_vectors)
            else:
                outputs[k] = output_layer(att_vectors, language, apply_mask)

        if len(outputs) == 1:
            # case with no MTL, return a tensor
            outputs = outputs[output_key[0]]

        # outputs: batch, unroll_steps, vocab_size
        return outputs, hidden, att_dict, att_vectors

    def init_hidden(self, encoder_final: Tensor) -> (Tensor, Optional[Tensor]):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        In case of `self.init_hidden_option == "bridge"`,
        this is a projection of the encoder state.

        In case of `self.init_hidden_option == "last"`
        and a size-matching `encoder_final`, this is set to the encoder state.
        If the encoder is twice as large as the decoder state (e.g. when
        bi-directional), just use the forward hidden state.

        In case of `self.init_hidden_option == "zero"`, it is initialized with
        zeros.

        For LSTMs we initialize both the hidden state and the memory cell
        with the same projection/copy of the encoder hidden state.

        All decoder layers are initialized with the same initial values.

        :param encoder_final: final state from the last layer of the encoder,
            shape (batch_size, encoder_hidden_size)
        :return: hidden state if GRU, (hidden state, memory cell) if LSTM,
            shape (batch_size, hidden_size)
        """
        batch_size, encoder_hidden_size = encoder_final.size()
        double_hidden = isinstance(self.rnn, nn.LSTM)

        if self.init_hidden_option == "zero":
            # return an appropriately-sized tensor of zeros
            hidden = encoder_final.new_zeros(batch_size, self.hidden_size)

        elif self.init_hidden_option == "bridge":
            assert self.bridge_layer is not None
            hidden = self.bridge_layer(encoder_final)  # batch x hidden_size

        else:
            hidden = encoder_final
            if encoder_hidden_size == 2 * self.hidden_size:
                # if encoder is bidirectional, use only forward state
                hidden = hidden[:, :self.hidden_size]

        # expand to n_layers x batch_size x hidden_size
        # The same hidden is used for all layers
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return (hidden, hidden) if double_hidden else hidden

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)


class MultiHeadRecurrentDecoder(Decoder):
    """A conditional RNN decoder with attention."""

    def __init__(self,
                 rnn_type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder_output_sizes: dict = None,
                 attention: str = "bahdanau",
                 attn_merge: str = "concat",
                 num_layers: int = 1,
                 dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 init_hidden: str = "bridge",
                 input_feeding: bool = True,
                 freeze: bool = False,
                 attn_func: str = "softmax",
                 attn_alpha: float = 1.5,
                 gate_func: str = "softmax",
                 gate_alpha: float = 1.5,
                 **kwargs) -> None:
        """
        Todo: document the unique challenges of making an RNN decoder that
        attends over multiple encoders
        """

        super(MultiHeadRecurrentDecoder, self).__init__(hidden_size, **kwargs)

        self.hidden_dropout = nn.Dropout(p=hidden_dropout)

        self.head_names = sorted(encoder_output_sizes)

        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        input_size = emb_size + hidden_size if input_feeding else emb_size

        # the decoder RNN
        self.rnn = rnn(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.)

        # combined output sizes of all encoders
        # this quantity matters for concat attention merging
        # it also matters if you have a bridge for init_hidden
        encoder_output_size = sum(encoder_output_sizes.values())

        assert attention in ["bahdanau", "luong"], \
            "Unknown attention mechanism: %s. Use 'bahdanau' or 'luong'."
        if attention == "bahdanau":
            attn_mechanism = partial(MultiAttention, query_size=hidden_size)
        else:
            attn_mechanism = MultiAttention
        self.attention = attn_mechanism(
            attn_type=attention,
            head_names=self.head_names,
            key_sizes=encoder_output_sizes,
            hidden_size=hidden_size,
            attn_func=attn_func,
            attn_alpha=attn_alpha,
            attn_merge=attn_merge,
            gate_func=gate_func,
            gate_alpha=gate_alpha)

        # to initialize from the final encoder state of last layer
        assert init_hidden == "bridge", \
            "only use bridge with multi-encoder models"

        self.bridge_layer = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size, bias=True),
            nn.Tanh()
        )

        if freeze:
            freeze_params(self)

    @property
    def rnn_input_size(self):
        return self.rnn.input_size

    @property
    def num_layers(self):
        return self.rnn.num_layers

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    def _check_shapes_input_forward_step(self,
                                         prev_embed: Tensor,
                                         prev_att_vector: Tensor,
                                         encoder_outputs: Dict[str, Tensor],
                                         masks: Dict[str, Tensor],
                                         hidden: Tensor) -> None:
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        """
        hidden_size = self.hidden_size
        encoder_output = encoder_outputs["src"]
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size([1, hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert all(m.size(0) == prev_embed.size(0) for m in masks.values())
        assert all(m.size(1) == 1 for m in masks.values())
        assert all(masks[k].size(2) == encoder_outputs[k].size(1)
                   for k in masks)
        if isinstance(hidden, tuple):  # for lstm
            hidden = hidden[0]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == hidden_size

    def _check_shapes_input_forward(self,
                                    trg_embed: Tensor,
                                    encoder_outputs: Dict[str, Tensor],
                                    masks: Dict[str, Tensor],
                                    hidden: Tensor = None,
                                    prev_att_vector: Tensor = None) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as `self.forward`.

        :param trg_embed:
        :param encoder_outputs:
        :param masks:
        :param hidden:
        :param prev_att_vector:
        """
        src_encoder_output, src_encoder_hidden, _ = encoder_outputs["src"]
        for k, v in encoder_outputs.items():
            encoder_output, encoder_hidden, _ = v
            assert len(encoder_output.shape) == 3
            assert len(encoder_hidden.shape) == 2
            assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
            assert masks[k].shape[0] == encoder_output.shape[0]
            assert masks[k].shape[2] == encoder_output.shape[1]
        assert all(v.size(1) == 1 for v in masks.values())

        assert trg_embed.shape[0] == src_encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):  # for lstm
                hidden = hidden[0]
            assert hidden.shape[1] == src_encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == src_encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(self,
                      prev_embed: Tensor,
                      prev_att_vector: Tensor,  # context or att vector
                      encoder_outputs: dict,
                      masks: dict,
                      hidden: Tensor) -> (Tensor, Tensor, Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
            shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
            shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
            shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
            shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
            shape (num_layers, batch_size, hidden_size)
        :return:
            - att_vector: new attention vector (batch_size, 1, hidden_size),
            - hidden: new hidden state with shape (batch_size, 1, hidden_size),
            - att_probs: attention probabilities (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(prev_embed=prev_embed,
                                              prev_att_vector=prev_att_vector,
                                              encoder_outputs=encoder_outputs,
                                              masks=masks,
                                              hidden=hidden)

        if self.input_feeding:
            # concatenate the input with the previous attention vector
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        # compute context vector using attention mechanism
        # only use last layer for attention mechanism
        # key projections are pre-computed
        srcs = {k: (encoder_outputs[k], masks[k]) for k in encoder_outputs}
        att_vector, att_probs = self.attention(query=query, srcs=srcs)

        att_vector = self.hidden_dropout(att_vector)

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs

    def forward(self,
                trg_embed: Tensor,
                encoder_outputs: dict,
                src_mask: Tensor,
                inflection_mask: Tensor,
                unroll_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None,
                **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Mostly the same as a normal RNN decoder, but with more masks.
        """

        enc_outputs = {k: v[0] for k, v in encoder_outputs.items()}
        enc_hiddens = {k: v[1] for k, v in encoder_outputs.items()}
        masks = {"src": src_mask, "inflection": inflection_mask}

        # shape checks
        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_outputs=encoder_outputs,
            masks=masks,
            hidden=hidden,
            prev_att_vector=prev_att_vector)

        # initialize decoder hidden state from final encoder hidden states
        if hidden is None:
            hidden = self.init_hidden(enc_hiddens)

        # pre-compute attention keys
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=enc_outputs)

        # store intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []
        att_probs = defaultdict(list)

        batch_size = trg_embed.size(0)
        if prev_att_vector is None:
            prev_att_vector = trg_embed.new_zeros(
                batch_size, 1, self.hidden_size
            )

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_outputs=enc_outputs,
                masks=masks,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            for k, prob in att_prob.items():
                att_probs[k].append(prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_probs = {k + "_trg": torch.cat(prob, dim=1)
                     for k, prob in att_probs.items()}
        # att_probs: batch, unroll_steps, src_length
        outputs = self.output_layers["vocab"](att_vectors)
        # outputs: batch, unroll_steps, vocab_size
        return outputs, hidden, att_probs, att_vectors

    def init_hidden(self, encoder_finals: Dict[str, Tensor]) \
            -> (Tensor, Optional[Tensor]):

        encoder_final = torch.cat(
            [encoder_finals[k] for k in self.head_names], dim=1
        )
        double_hidden = isinstance(self.rnn, nn.LSTM)

        assert self.bridge_layer is not None
        hidden = self.bridge_layer(encoder_final)  # batch x hidden_size

        # expand to n_layers x batch_size x hidden_size
        # The same hidden is used for all layers
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return (hidden, hidden) if double_hidden else hidden

    def __repr__(self):
        # update the representation
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """
    layer_module = TransformerDecoderLayer

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 freeze: bool = False,
                 self_attn_func: str = "softmax",
                 src_attn_func: str = "softmax",
                 self_attn_alpha: float = 1.5,
                 src_attn_alpha: float = 1.5,
                 merge: str = "serial",  # for multi-encoder models
                 gate_func: str = "softmax",
                 gate_alpha: float = 1.5,
                 **kwargs):
        """
        Initialize a Transformer decoder.

        :param num_layers:
        :param num_heads:
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout:
        :param emb_dropout: dropout probability for embeddings
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs: passed to generic Decoder Constructor
        """
        super(TransformerDecoder, self).__init__(hidden_size, **kwargs)

        self.layers = nn.ModuleList(
            [self.layer_module(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                self_attn_func=self_attn_func,
                self_attn_alpha=self_attn_alpha,
                src_attn_func=src_attn_func,
                src_attn_alpha=src_attn_alpha,
                merge=merge,
                gate_func=gate_func,
                gate_alpha=gate_alpha)
             for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        if freeze:
            freeze_params(self)

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def num_heads(self):
        return self.layers[0].trg_trg_att.num_heads

    def _forward_pre_output(self,
                            trg_embed: Tensor = None,
                            encoder_output: Tensor = None,
                            src_mask: Tensor = None,
                            trg_mask: Tensor = None,
                            **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :return:
            context attn probs: batch x layer x head x tgt x src
        """
        assert trg_mask is not None, "trg_mask is required for Transformer"

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        self_attn_layers = []
        context_attn_layers = []
        for layer in self.layers:
            x, self_attn, ctx_attn = layer(
                x=x, memory=encoder_output,
                src_mask=src_mask, trg_mask=trg_mask
            )
            self_attn_layers.append(self_attn)
            context_attn_layers.append(ctx_attn)

        x = self.layer_norm(x)

        trg_trg_attn = torch.stack(self_attn_layers, dim=1)
        src_trg_attn = torch.stack(context_attn_layers, dim=1)
        attn = {"trg_trg": trg_trg_attn, "src_trg": src_trg_attn}

        return x, attn, None

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                language: Tensor = None,
                output_key: str = "vocab",
                apply_mask: bool = True,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :return:
            context attn probs: batch x layer x head x tgt x src
        """

        x, attn, _ = self._forward_pre_output(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            src_mask=src_mask,
            trg_mask=trg_mask,
            **kwargs)

        outputs = dict()
        if isinstance(output_key, str):
            output_key = [output_key]
        for k in output_key:
            output_layer = self.output_layers[k]
            if isinstance(output_layer, nn.Linear):
                outputs[k] = output_layer(x)
            else:
                outputs[k] = output_layer(x, language, apply_mask)

        if len(outputs) == 1:
            # case with no MTL, return a tensor
            outputs = outputs[output_key[0]]

        return outputs, x, attn, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, self.num_layers, self.num_heads)


class MultiSourceTransformerDecoder(TransformerDecoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """
    layer_module = MultiSourceTransformerDecoderLayer

    def __init__(self, *args, **kwargs):
        """
        Initialize a multi-source transformer decoder
        """
        super(MultiSourceTransformerDecoder, self).__init__(*args, **kwargs)

    def forward(self,
                trg_embed: Tensor = None,
                encoder_outputs: dict = None,  # note the change
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                inflection_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :return:
            context attn probs: batch x layer x head x tgt x src
        """
        assert trg_mask is not None, "trg_mask is required for Transformer"

        # encoder_outputs is not an ideal name for this dict because it also
        # contains encoder_hidden, which transformers do not use
        enc_outputs = {k: v[0] for k, v in encoder_outputs.items()}

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        self_attn_layers = []
        context_attn_layers = defaultdict(list)
        for layer in self.layers:
            # todo: multiple (variable?) numbers of context attentions
            # return a dictionary of context attentions
            x, self_attn, ctx_attn = layer(
                x=x, memories=enc_outputs,
                src_mask=src_mask, trg_mask=trg_mask,
                inflection_mask=inflection_mask
            )
            self_attn_layers.append(self_attn)
            for enc_name, enc_ctx_attn in ctx_attn.items():
                context_attn_layers[enc_name].append(enc_ctx_attn)

        # todo: inflection_trg attention
        trg_trg_attn = torch.stack(self_attn_layers, dim=1)
        attn = {"trg_trg": trg_trg_attn}
        for enc_name, enc_ctx_attn_layers in context_attn_layers.items():
            enc_stacked_attn = torch.stack(enc_ctx_attn_layers, dim=1)
            attn[enc_name + "_trg"] = enc_stacked_attn

        x = self.layer_norm(x)
        output = self.output_layers["vocab"](x)

        return output, x, attn, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, self.num_layers, self.num_heads)
