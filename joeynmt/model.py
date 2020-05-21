# coding: utf-8
"""
Module to represents whole models
"""

from typing import Dict
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings, MultispaceEmbeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder, \
    MultiHeadRecurrentDecoder, MultiSourceTransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch
from joeynmt.helpers import ConfigurationError, tile


def pad_and_stack_hyps(hyps, pad_value):
    shape = len(hyps), max(h.shape[0] for h in hyps)
    filled = torch.full(shape, pad_value, dtype=int)
    for j, h in enumerate(hyps):
        filled[j, :h.shape[0]] = h
    return filled


class _Model(nn.Module):

    def __init__(self):
        super(_Model, self).__init__()

    def encode(self, batch):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def _transformer(self):
        return isinstance(self.decoder, TransformerDecoder)

    def forward(self, batch, output_key: str = "vocab", **kwargs):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param batch:
        :param output_key: for multi-task models
        :return: decoder outputs
        """

        encoder_outputs = self.encode(batch)

        unroll_steps = batch.trg_input.size(1)
        return self.decode(encoder_outputs=encoder_outputs,
                           src_mask=batch.src_mask,
                           trg_input=batch.trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=batch.trg_mask,
                           output_key=output_key,
                           language=batch.language,
                           inflection_mask=batch.inflection_mask,
                           generate=False,
                           **kwargs)

    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  scorer, log_sparsity=False, apply_mask=False):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        with torch.no_grad():
            encoder_outputs = self.encode(batch)

            # if maximum output length is not specified, adapt to src len
            if max_output_length is None:
                max_output_length = int(batch.src_lengths.max().item() * 1.5)

            # greedy decoding
            if beam_size == 0:
                stacked_output, dec_attn, probs = self._greedy(
                    encoder_outputs=encoder_outputs,
                    language=batch.language,
                    src_mask=batch.src_mask,
                    inflection_mask=batch.inflection_mask,
                    max_output_length=max_output_length,
                    log_sparsity=log_sparsity,
                    apply_mask=apply_mask
                )
            else:
                stacked_output, dec_attn, probs = self._beam_search(
                    size=beam_size,
                    encoder_outputs=encoder_outputs,
                    language=batch.language,
                    src_mask=batch.src_mask,
                    inflection_mask=batch.inflection_mask,
                    max_output_length=max_output_length,
                    scorer=scorer,
                    apply_mask=apply_mask
                )

            attn = dict()
            enc_attn = self._enc_attn(encoder_outputs)
            if enc_attn is not None:
                for k, v in enc_attn.items():
                    attn[k] = v.cpu().numpy()
            if dec_attn is not None:
                for k, v in dec_attn.items():
                    attn[k] = v
            return stacked_output, attn, probs

    def _enc_attn(self, encoder_outputs):
        """
        encoder_outputs: whatever is returned by encode()
        return a dict, or None
        """
        raise NotImplementedError

    def _greedy(self, encoder_outputs, src_mask, max_output_length,
                log_sparsity, **kwargs):
        """
        Greedily decode from the model
        """

        batch_size = src_mask.size(0)
        prev_y = src_mask.new_full(
            [batch_size, 1], self.bos_index, dtype=torch.long
        )

        output = []
        dists = []
        attn_scores = defaultdict(list)
        hidden = None
        prev_att_vector = None

        trg_mask = src_mask.new_ones([1, 1, 1]) if self._transformer else None
        finished = src_mask.new_zeros((batch_size, 1)).bool()

        for t in range(max_output_length):

            # decode one single step
            dec_out, hidden, att_probs, prev_att_vector = self.decode(
                encoder_outputs=encoder_outputs,
                trg_input=prev_y,
                src_mask=src_mask,
                trg_mask=trg_mask,
                decoder_hidden=hidden,
                prev_att_vector=prev_att_vector,
                unroll_steps=1,
                output_key="vocab",
                **kwargs)

            # logits: batch x time x vocab
            logits = dec_out[:, -1]

            if log_sparsity:
                dists.append(logits)

            # argmax makes it greedy
            next_word = torch.argmax(logits, dim=-1).unsqueeze(1)

            if self._transformer:
                # transformer: keep all previous steps as input to decoder
                prev_y = torch.cat([prev_y, next_word], dim=1)
            else:
                # rnn: only the most recent step is input to the decoder
                prev_y = next_word
                output.append(next_word.squeeze(1).cpu().numpy())
                for k, v in att_probs.items():
                    attn_scores[k].append(v.squeeze(1).cpu().numpy())

            is_eos = torch.eq(next_word, self.eos_index)
            finished += is_eos
            # stop predicting if <eos> reached for all of batch
            if (finished >= 1).sum() == batch_size:
                break

        stacked_output = prev_y[:, 1:] if self._transformer \
            else np.stack(output, axis=1)

        # batch x len x V
        if log_sparsity:
            assert self.decoder.gen_func is not None
            probs = self.decoder.gen_func(
                torch.stack(dists)
            ).transpose(0, 1).contiguous()
        else:
            probs = None

        if self._transformer:
            stacked_attn = {k: v.cpu().numpy() for k, v in att_probs.items()}
        else:
            stacked_attn = {k: np.stack(v, axis=1) for k, v in attn_scores.items()}
        return stacked_output, stacked_attn, probs

    def _beam_search(
            self,
            size: int,
            src_mask: Tensor,
            max_output_length: int,
            scorer,
            encoder_outputs,
            n_best: int = 1,
            language: Tensor = None,
            inflection_mask: Tensor = None,
            **kwargs):
        """
        Beam search with size k.
        Inspired by OpenNMT-py, adapted for Transformer.

        In each decoding step, find the k most likely partial hypotheses.

        :param size: size of the beam
        :param encoder_outputs: whatever is returned by encode()
        :param src_mask:
        :param max_output_length:
        :param n_best: return this many hypotheses, <= beam
        :return:
            - stacked_output: output hypotheses (2d array of indices),
            - stacked_attention_scores: attention scores (3d array)
        """
        assert n_best == 1

        batch_size = src_mask.size(0)
        att_vectors = None  # not used for Transformer
        device = src_mask.device

        # Recurrent models only: initialize RNN hidden state
        if not self._transformer:
            hidden = self._init_decoder_hidden(encoder_outputs)
        else:
            hidden = None

        # tile encoder states and decoder initial states beam_size times
        if hidden is not None:
            # layers x batch*k x dec_hidden_size
            if isinstance(hidden, list):
                hidden = [tile(h, size, dim=1) for h in hidden]
            else:
                hidden = tile(hidden, size, dim=1)

        if language is not None:
            language = tile(language, size, dim=0)

        encoder_outputs = self._tile_encoder_outputs(encoder_outputs, size)

        src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len
        # block for subclass
        if inflection_mask is not None:
            inflection_mask = tile(inflection_mask, size, dim=0)

        # Transformer only: create target mask
        trg_mask = src_mask.new_ones([1, 1, 1]) if self._transformer else None

        # numbering elements in the batch
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

        # numbering elements in the extended batch, i.e. beam size copies of
        # each batch element
        beam_offset = torch.arange(
            0,
            batch_size * size,
            step=size,
            dtype=torch.long,
            device=device)

        # keeps track of the top beam size hypotheses to expand for each
        # element in the batch to be further decoded (that are still "alive")
        alive_seq = torch.full(
            (batch_size * size, 1),
            self.bos_index,
            dtype=torch.long,
            device=device
        )

        # Give full probability to the first beam on the first step.
        current_beam = torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                    device=device).repeat(batch_size, 1)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(1, max_output_length + 1):
            # Transformers need the complete predicted sentence so far.
            # Recurrent models only need the last target word prediction
            if self._transformer:
                dec_input = alive_seq
            else:
                dec_input = alive_seq[:, -1].view(-1, 1)

            # decode a single step
            # log_probs: batch*k x trg_vocab
            log_probs, hidden, _, att_vectors = self.decode(
                encoder_outputs=encoder_outputs,
                trg_input=dec_input,
                src_mask=src_mask,
                inflection_mask=inflection_mask,
                decoder_hidden=hidden,
                prev_att_vector=att_vectors,
                unroll_steps=1,
                trg_mask=trg_mask,
                language=language,
                generate=True,
                **kwargs
            )

            # multiply probs by the beam probability (=add logprobs)
            raw_scores = log_probs + current_beam.view(-1).unsqueeze(1)

            # flatten log_probs into a list of possibilities
            raw_scores = raw_scores.reshape(-1, size * len(self.trg_vocab))

            # apply an additional scorer, such as a length penalty
            scores = scorer(raw_scores, step) if scorer is not None \
                else raw_scores

            # pick currently best top k hypotheses (flattened order)
            topk_scores, topk_ids = scores.topk(size, dim=-1)

            # scores are distinct from log probabilities if using a length
            # penalty. The beam will contain the log probabilities regardless,
            # so they need to be recovered
            current_beam = topk_scores if scorer is None \
                else raw_scores.gather(1, topk_ids)

            # reconstruct beam origin and true word ids from flattened order
            topk_beam_index = topk_ids.div(len(self.trg_vocab))
            topk_ids = topk_ids.fmod(len(self.trg_vocab))

            # map beam_index to batch_index in the flat representation
            b_off = beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
            batch_index = topk_beam_index + b_off
            select_ix = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_ix), topk_ids.view(-1, 1)],
                -1
            )  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(self.eos_index)
            if step == max_output_length:
                is_finished.fill_(1)
            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(1)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # store finished hypotheses for this batch
                    for j in finished_hyp:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:])
                        )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                unfinished = end_condition.eq(0).nonzero().view(-1)
                # if all sentences are translated, no need to go further
                # pylint: disable=len-as-condition
                if len(unfinished) == 0:
                    break
                # remove finished batches for the next step
                current_beam = current_beam.index_select(0, unfinished)
                batch_index = batch_index.index_select(0, unfinished)
                batch_offset = batch_offset.index_select(0, unfinished)
                alive_seq = predictions.index_select(0, unfinished) \
                    .view(-1, alive_seq.size(-1))

            # reorder indices, outputs and masks
            select_ix = batch_index.view(-1)

            encoder_outputs = self._select_encoder_ix(
                encoder_outputs, select_ix
            )

            src_mask = src_mask.index_select(0, select_ix)
            if inflection_mask is not None:
                inflection_mask = inflection_mask.index_select(0, select_ix)

            hidden = self._select_hidden_ix(hidden, select_ix)

            if att_vectors is not None:
                # I've been handling other stuff like this through subclassing
                # maybe this should be part of a _select_unfinished or sthing
                if isinstance(att_vectors, list):
                    att_vectors = [av.index_select(0, select_ix) if av is not None else None
                                   for av in att_vectors]
                else:
                    att_vectors = att_vectors.index_select(0, select_ix)

            if language is not None:
                language = language.index_select(0, select_ix)

        final_outputs = pad_and_stack_hyps(
            [r[0].cpu() for r in results["predictions"]], self.pad_index
        )
        return final_outputs, None, None

    def _select_hidden_ix(self, hidden, select_ix):
        if self._transformer or hidden is None:
            return None
        if isinstance(hidden, tuple):
            # for LSTMs, states are tuples of tensors
            h, c = hidden
            h = h.index_select(1, select_ix)
            c = c.index_select(1, select_ix)
            hidden = h, c
        else:
            # for GRUs, states are single tensors
            hidden = hidden.index_select(1, select_ix)
        return hidden

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module,
                           language_loss: nn.Module = None,
                           language_weight: float = 0.0,
                           apply_mask: bool = False) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        assert language_loss is None or language_weight != 0.0

        output_keys = ["vocab"]
        if language_loss is not None:
            output_keys.append("language")

        out, hidden, att_probs, _ = self(
            batch, output_key=output_keys, apply_mask=apply_mask)

        vocab_out = out if isinstance(out, Tensor) else out["vocab"]

        vocab_out = vocab_out.view(-1, vocab_out.size(-1))

        # compute batch loss
        gold_trg = batch.trg.contiguous().view(-1)

        main_loss = loss_function(vocab_out, gold_trg)

        # return batch loss = sum over all elements in batch that are not pad
        if language_loss is not None:
            lang_out = out["language"]
            lang_gold = batch.language.unsqueeze(1).expand(
                -1, lang_out.size(1)
            ).contiguous().view(-1)
            lang_out = lang_out.view(-1, lang_out.size(-1))
            lang_loss = language_weight * language_loss(lang_out, lang_gold)
            return main_loss + lang_loss

        return main_loss


class Model(_Model):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    def encode(self, batch) -> (Tensor, Tensor, Tensor):
        """
        Encodes the source sentence.

        :param batch:
        :return: encoder outputs (output, hidden_concat)
        """
        src_emb = self.src_embed(batch.src, language=batch.language)
        return self.encoder(src_emb, batch.src_lengths, batch.src_mask)

    def decode(self,
               encoder_outputs: tuple,
               src_mask: Tensor,
               trg_input: Tensor,
               unroll_steps: int,
               decoder_hidden: Tensor = None,
               trg_mask: Tensor = None,
               language: Tensor = None,
               generate: bool = False,
               **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_outputs: encoder states for decoder initialization and
            attention computation
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        encoder_output, encoder_hidden, _ = encoder_outputs
        trg_emb = self.trg_embed(trg_input, language=language)
        dec_out, hidden, att_probs, prev_att_vector = self.decoder(
            trg_embed=trg_emb,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            trg_mask=trg_mask,
            language=language,
            **kwargs
        )
        if generate:
            dec_out = self.decoder.gen_func(dec_out[:, -1], dim=-1).squeeze(1)
        return dec_out, hidden, att_probs, prev_att_vector

    def _enc_attn(self, encoder_outputs):
        """
        encoder_outputs: whatever is returned by encode()
        return a dict, or None
        """
        return encoder_outputs[2]  # single encoder

    def _init_decoder_hidden(self, encoder_outputs):
        encoder_hidden = encoder_outputs[1]
        return self.decoder.init_hidden(encoder_hidden)

    def _tile_encoder_outputs(self, encoder_outputs, size):
        encoder_output, encoder_hidden, _ = encoder_outputs
        encoder_output = tile(encoder_output.contiguous(), size, dim=0)
        return encoder_output, encoder_hidden, None

    def _select_encoder_ix(self, encoder_outputs, select_ix):
        encoder_output, encoder_hidden, _ = encoder_outputs
        encoder_output = encoder_output.index_select(0, select_ix)
        return encoder_output, encoder_hidden, None

    def __repr__(self):
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)


class MultiEncoderModel(_Model):

    def __init__(self,
                 encoders: Dict[str, Encoder],
                 decoder: Decoder,
                 enc_embeds: Dict[str, Embeddings],
                 trg_embed: Embeddings,
                 vocabs: Dict[str, Vocabulary]):
        """
        Create a multi-encoder seq2seq model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        assert set(enc_embeds) == set(encoders)
        assert set(enc_embeds) < set(vocabs)
        super(MultiEncoderModel, self).__init__()

        self.enc_embeds = nn.ModuleDict(enc_embeds)
        self.trg_embed = trg_embed
        self.encoders = nn.ModuleDict(encoders)
        self.decoder = decoder
        self.vocabs = vocabs
        self.bos_index = vocabs["trg"].stoi[BOS_TOKEN]
        self.pad_index = vocabs["trg"].stoi[PAD_TOKEN]
        self.eos_index = vocabs["trg"].stoi[EOS_TOKEN]

    @property
    def src_vocab(self):
        return self.vocabs["src"]

    @property
    def trg_vocab(self):
        return self.vocabs["trg"]

    @property
    def encoder(self):
        return self.encoders["src"]

    @property
    def src_embed(self):
        return self.enc_embeds["src"]

    def encode(self, batch):
        """
        Encode a batch with fields for multiple encoders. At the moment, this
        only supports the case where there are two encoder fields, called
        "src" and "inflection". Future work will remove these magic words.
        """
        outputs = dict()
        src_embed = self.enc_embeds["src"](batch.src, language=batch.language)
        src_output = self.encoders["src"](
            src_embed, batch.src_lengths, batch.src_mask
        )
        outputs["src"] = src_output

        inflection_embed = self.enc_embeds["inflection"](
            batch.inflection, language=batch.language
        )
        inflection_output = self.encoders["inflection"](
            inflection_embed, batch.inflection_lengths, batch.inflection_mask
        )
        outputs["inflection"] = inflection_output

        return outputs

    def decode(self,
               encoder_outputs: dict,
               src_mask: Tensor,
               inflection_mask: Tensor,
               trg_input: Tensor,
               unroll_steps: int,
               decoder_hidden: Tensor = None,
               trg_mask: Tensor = None,
               language: Tensor = None,
               generate: bool = False,
               **kwargs) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_outputs: encoder states for attention computation and
            decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        trg_embed = self.trg_embed(trg_input, language=language)
        dec_out, hidden, att_probs, prev_att_vector = self.decoder(
            trg_embed=trg_embed,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask,
            inflection_mask=inflection_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            trg_mask=trg_mask,
            language=language,
            **kwargs)
        if generate:
            dec_out = self.decoder.gen_func(dec_out[:, -1], dim=-1).squeeze(1)
        return dec_out, hidden, att_probs, prev_att_vector

    @classmethod
    def _enc_attn(cls, encoder_outputs):
        """
        encoder_outputs: whatever is returned by encode()
        return a dict, or None
        """
        return None  # you can think about this if you want attn plots

    def _init_decoder_hidden(self, encoder_outputs):
        encoder_hidden = {k: v[1] for k, v in encoder_outputs.items()}
        return self.decoder.init_hidden(encoder_hidden)

    def _tile_encoder_outputs(self, encoder_outputs, size):
        result = dict()
        for k, (encoder_output, encoder_hidden, _) in encoder_outputs.items():
            encoder_output = tile(encoder_output.contiguous(), size, dim=0)
            result[k] = (encoder_output, encoder_hidden, None)
        return result

    def _select_encoder_ix(self, encoder_outputs, select_ix):
        result = dict()
        for k in encoder_outputs:
            enc_out, enc_hid, _ = encoder_outputs[k]
            enc_out = enc_out.index_select(0, select_ix)
            result[k] = (enc_out, enc_hid, None)
        return result


class EnsembleModel(_Model):

    def __init__(self, *models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        assert(all(m.src_vocab == self.src_vocab for m in self.models))
        assert(all(m.trg_vocab == self.trg_vocab for m in self.models))
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    @property
    def src_vocab(self):
        return self.models[0].src_vocab

    @property
    def trg_vocab(self):
        return self.models[0].trg_vocab

    def encode(self, batch):
        return [m.encode(batch) for m in self.models]

    @property
    def _transformer(self):
        return any(isinstance(m.decoder, TransformerDecoder)
                   for m in self.models)

    def _enc_attn(self, encoder_outputs):
        return None

    def _init_decoder_hidden(self, encoder_outputs):
        return [m._init_decoder_hidden(enc_outs)
                for m, enc_outs in zip(self.models, encoder_outputs)]

    def _tile_encoder_outputs(self, encoder_outputs, size):
        return [m._tile_encoder_outputs(enc_outs, size)
                for m, enc_outs in zip(self.models, encoder_outputs)]

    def _select_encoder_ix(self, encoder_outputs, select_ix):
        return [m._select_encoder_ix(enc_outs, select_ix)
                for m, enc_outs in zip(self.models, encoder_outputs)]

    def _select_hidden_ix(self, hidden, select_ix):
        return [m._select_hidden_ix(h, select_ix)
                for m, h in zip(self.models, hidden)]

    def decode(self,
               encoder_outputs: list,
               src_mask: Tensor,
               trg_input: Tensor,
               unroll_steps: int,
               prev_att_vector: list = None,
               decoder_hidden: list = None,
               trg_mask: Tensor = None,
               language: Tensor = None,
               generate: bool = False,
               **kwargs) -> (Tensor, Tensor, Tensor, Tensor):

        assert len(encoder_outputs) == len(self.models)
        if prev_att_vector is not None:
            assert len(prev_att_vector) == len(self.models)
        else:
            prev_att_vector = [None] * len(self.models)
        if decoder_hidden is not None:
            assert len(decoder_hidden) == len(self.models)
        else:
            decoder_hidden = [None] * len(self.models)
        single_decodes = []
        inputs = zip(self.models, encoder_outputs,
                     prev_att_vector, decoder_hidden)
        for model, enc_outs, prev_att, dec_hid in inputs:
            single_decode = model.decode(
                encoder_outputs=enc_outs,
                src_mask=src_mask,
                trg_input=trg_input,
                unroll_steps=unroll_steps,
                decoder_hidden=dec_hid,
                prev_att_vector=prev_att,
                trg_mask=trg_mask,
                language=language,
                generate=generate,  # maybe should require True
                **kwargs
            )
            single_decodes.append(single_decode)

        dec_outs = [d[0] for d in single_decodes]
        hiddens = [d[1] for d in single_decodes]
        att_vectors = [d[3] for d in single_decodes]
        # now, gonna assume that the individual models returned log probs
        # dimensions n x V
        log_probs = torch.log(torch.exp(torch.stack(dec_outs)).mean(dim=0))
        return log_probs, hiddens, None, att_vectors


def build_embeddings(emb_config: dict, vocab: Vocabulary):
    padding_idx = vocab.stoi[PAD_TOKEN]

    embed = Embeddings(
        **emb_config, vocab_size=len(vocab), padding_idx=padding_idx
    )
    return embed


def build_multispace_embeddings(emb_configs: dict, vocabs: dict, main: str):
    assert set(emb_configs) <= set(vocabs)
    assert main in emb_configs
    embs = {k: build_embeddings(emb_configs[k], vocabs[k])
            for k in emb_configs}
    main_emb = embs.pop(main)
    return MultispaceEmbeddings(main_emb, mode="feature", **embs)


def build_encoder(config: dict, emb_size: int):
    dropout = config.get("dropout", 0.)
    emb_config = config["embeddings"]
    emb_dropout = emb_config.get("dropout", dropout)
    enc_type = config.get("type", "recurrent")
    if enc_type == "transformer":
        assert emb_size == config["hidden_size"], \
            "for transformer, emb_size must be hidden_size"
    enc_class = TransformerEncoder if enc_type == "transformer" \
        else RecurrentEncoder
    encoder = enc_class(
        **config, emb_size=emb_size, emb_dropout=emb_dropout
    )
    return encoder


def build_model(cfg: dict = None, vocabs: dict = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    src_vocab = vocabs["src"]
    trg_vocab = vocabs["trg"]
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    if "encoder" in cfg:
        enc_configs = {"src": cfg["encoder"]}
    else:
        assert "encoders" in cfg
        enc_configs = cfg["encoders"]

    enc_embeds = dict()
    encoders = dict()
    for name, enc_config in enc_configs.items():
        vocab = vocabs[name]
        if "multispace_embeddings" in enc_config:
            # multispace embeddings (meaning with a language feature)
            src_embed = build_multispace_embeddings(
                enc_config["multispace_embeddings"], vocabs, name)
        else:
            src_embed = build_embeddings(enc_config["embeddings"], vocab)

        encoder = build_encoder(enc_config, src_embed.embedding_dim)
        enc_embeds[name] = src_embed
        encoders[name] = encoder

    multi_encoder = len(encoders) > 1

    dec_config = cfg["decoder"]

    # this ties source and target embeddings
    if cfg.get("tied_embeddings", False):
        assert vocabs["src"].itos == trg_vocab.itos, \
            "Embedding cannot be tied because vocabularies differ."
        trg_embed = enc_embeds["src"]
    else:
        if "multispace_embeddings" in dec_config:
            trg_embed = build_multispace_embeddings(
                dec_config["multispace_embeddings"], vocabs, "trg")
        else:
            trg_embed = build_embeddings(dec_config["embeddings"], trg_vocab)

    # build decoder
    dec_dropout = dec_config.get("dropout", 0.)
    if "embeddings" not in dec_config:
        dec_emb_dropout = dec_dropout
    else:
        dec_emb_dropout = dec_config["embeddings"].get("dropout", dec_dropout)
    dec_type = dec_config.get("type", "recurrent")

    if not multi_encoder:
        dec_class = TransformerDecoder if dec_type == "transformer" \
            else RecurrentDecoder
    else:
        enc_out_sizes = {n: enc.output_size for n, enc in encoders.items()}
        if dec_type == "transformer":
            dec_class = MultiSourceTransformerDecoder
        else:
            dec_class = partial(
                MultiHeadRecurrentDecoder, encoder_output_sizes=enc_out_sizes
            )

    decoder = dec_class(
        **dec_config,
        encoder_output_size=encoders["src"].output_size,
        vocab_size=len(trg_vocab),
        emb_size=trg_embed.embedding_dim,
        emb_dropout=dec_emb_dropout)

    if not multi_encoder:
        model = Model(
            encoder=encoders["src"],
            decoder=decoder,
            src_embed=enc_embeds["src"],
            trg_embed=trg_embed,
            src_vocab=vocabs["src"],
            trg_vocab=trg_vocab)
    else:
        model = MultiEncoderModel(
            encoders=encoders,
            decoder=decoder,
            enc_embeds=enc_embeds,
            trg_embed=trg_embed,
            vocabs=vocabs)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        assert isinstance(model.decoder.output_layers["vocab"], nn.Linear)
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layers["vocab"].weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layers["vocab"].weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
