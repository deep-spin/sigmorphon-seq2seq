# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
import logging
from typing import List, Optional, Sequence
from functools import partial
from collections import defaultdict
import numpy as np

import torch
from torchtext.data import Dataset, Field

from joeynmt.helpers import bpe_postprocess, load_config, \
    get_latest_checkpoint, load_checkpoint, store_attention_plots, \
    ConfigurationError
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy, \
    character_error_rate, wer
from joeynmt.model import build_model, Model, EnsembleModel
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter, MonoDataset
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary


def len_penalty(input, step, alpha):
    length_penalty = ((5.0 + step) / 6.0) ** alpha
    return input / length_penalty


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, data: Dataset,
                     batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     src_level: str,
                     trg_level: str,
                     eval_metrics: Optional[Sequence[str]],
                     attn_metrics: Optional[Sequence[str]],
                     loss_function: torch.nn.Module = None,
                     beam_size: int = 0, beam_alpha: int = 0,
                     batch_type: str = "sentence",
                     save_attention: bool = False,
                     log_sparsity: bool = False,
                     apply_mask: bool = True  # hmm
                     ) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param src_level: source segmentation level, one of "char", "bpe", "word"
    :param trg_level: target segmentation level, one of "char", "bpe", "word"
    :param eval_metrics: evaluation metric, e.g. "bleu"
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If 0 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to 0 (default).
    :param batch_type: validation batch type (sentence or token)

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    eval_funcs = {
        "bleu": bleu,
        "chrf": chrf,
        "token_accuracy": partial(token_accuracy, level=trg_level),
        "sequence_accuracy": sequence_accuracy,
        "wer": wer,
        "cer": partial(character_error_rate, level=trg_level)
    }
    selected_eval_metrics = {name: eval_funcs[name] for name in eval_metrics}

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)
    valid_sources_raw = [s for s in data.src]
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    scorer = partial(len_penalty, alpha=beam_alpha) if beam_alpha > 0 else None
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = defaultdict(list)
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        total_attended = defaultdict(int)
        greedy_steps = 0
        greedy_supported = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_lengths()

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, loss_function=loss_function)
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            output, attention_scores, probs = model.run_batch(
                batch=batch, beam_size=beam_size, scorer=scorer,
                max_output_length=max_output_length, log_sparsity=log_sparsity,
                apply_mask=apply_mask)
            if log_sparsity:
                lengths = torch.LongTensor((output == model.trg_vocab.stoi[EOS_TOKEN]).argmax(axis=1)).unsqueeze(1)
                batch_greedy_steps = lengths.sum().item()
                greedy_steps += lengths.sum().item()

                ix = torch.arange(output.shape[1]).unsqueeze(0).expand(output.shape[0], -1)
                mask = ix <= lengths
                supp = probs.exp().gt(0).sum(dim=-1).cpu()  # batch x len
                supp = torch.where(mask, supp, torch.tensor(0)).sum()
                greedy_supported += supp.float().item()

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])

            if attention_scores is not None:
                # is attention_scores ever None?
                if save_attention:
                    # beam search currently does not support attention logging
                    for k, v in attention_scores.items():
                        valid_attention_scores[k].extend(v[sort_reverse_index])
                if attn_metrics:
                    # add to total_attended
                    for k, v in attention_scores.items():
                        total_attended[k] += (v > 0).sum()

        assert len(all_outputs) == len(data)

        if log_sparsity:
            print(greedy_supported / greedy_steps)

        valid_scores = dict()
        if loss_function is not None and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            valid_scores["loss"] = total_loss
            valid_scores["ppl"] = torch.exp(total_loss / total_ntokens)

        # decode back to symbols
        decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                            cut_at_eos=True)

        # evaluate with metric on full dataset
        src_join_char = " " if src_level in ["word", "bpe"] else ""
        trg_join_char = " " if trg_level in ["word", "bpe"] else ""
        valid_sources = [src_join_char.join(s) for s in data.src]
        valid_references = [trg_join_char.join(t) for t in data.trg]
        valid_hypotheses = [trg_join_char.join(t) for t in decoded_valid]

        if attn_metrics:
            decoded_ntokens = sum(len(t) for t in decoded_valid)
            for attn_metric in attn_metrics:
                assert attn_metric == "support"
                for attn_name, tot_attended in total_attended.items():
                    score_name = attn_name + "_" + attn_metric
                    # this is not the right denominator
                    valid_scores[score_name] = tot_attended / decoded_ntokens

        # post-process
        if src_level == "bpe":
            valid_sources = [bpe_postprocess(s) for s in valid_sources]
        if trg_level == "bpe":
            valid_references = [bpe_postprocess(v) for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v) for v in valid_hypotheses]

        languages = [language for language in data.language]
        by_language = defaultdict(list)
        seqs = zip(valid_references, valid_hypotheses) if valid_references else valid_hypotheses
        if languages:
            examples = zip(languages, seqs)
            for lang, seq in examples:
                by_language[lang].append(seq)
        else:
            by_language[None].extend(seqs)

        # if references are given, evaluate against them
        # incorrect if-condition?
        # scores_by_lang = {name: dict() for name in selected_eval_metrics}
        scores_by_lang = dict()
        if valid_references and eval_metrics is not None:
            assert len(valid_hypotheses) == len(valid_references)

            for eval_metric, eval_func in selected_eval_metrics.items():
                score_by_lang = dict()
                for lang, pairs in by_language.items():
                    lang_hyps, lang_refs = zip(*pairs)
                    lang_score = eval_func(lang_hyps, lang_refs)
                    score_by_lang[lang] = lang_score

                score = sum(score_by_lang.values()) / len(score_by_lang)
                valid_scores[eval_metric] = score
                scores_by_lang[eval_metric] = score_by_lang

    if not languages:
        scores_by_lang = None
    return valid_scores, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        decoded_valid, valid_attention_scores, scores_by_lang, by_language


# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt,  # str or list now
         output_path: str = None,
         save_attention: bool = False,
         logger: logging.Logger = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        FORMAT = '%(asctime)-15s - %(message)s'
        logging.basicConfig(format=FORMAT)
        logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    test_cfg = cfg["testing"]

    if "test" not in data_cfg.keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = train_cfg["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    batch_size = train_cfg.get("eval_batch_size", train_cfg["batch_size"])
    batch_type = train_cfg.get("eval_batch_type", train_cfg.get("batch_type", "sentence"))
    use_cuda = train_cfg.get("use_cuda", False)
    src_level = data_cfg.get("src_level", data_cfg.get("level", "word"))
    trg_level = data_cfg.get("trg_level", data_cfg.get("level", "word"))

    eval_metric = train_cfg["eval_metric"]
    if isinstance(eval_metric, str):
        eval_metric = [eval_metric]
    attn_metric = train_cfg.get("attn_metric", [])
    if isinstance(attn_metric, str):
        attn_metric = [attn_metric]
    max_output_length = train_cfg.get("max_output_length", None)

    # load the data
    data = load_data(data_cfg)
    dev_data = data["dev_data"]
    test_data = data["test_data"]
    vocabs = data["vocabs"]

    data_to_predict = {"dev": dev_data, "test": test_data}

    # load model state from disk
    if isinstance(ckpt, str):
        ckpt = [ckpt]
    individual_models = []
    for c in ckpt:
        model_checkpoint = load_checkpoint(c, use_cuda=use_cuda)

        # build model and load parameters into it
        m = build_model(cfg["model"], vocabs=vocabs)
        m.load_state_dict(model_checkpoint["model_state"])
        individual_models.append(m)
    if len(individual_models) == 1:
        model = individual_models[0]
    else:
        model = EnsembleModel(*individual_models)

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_sizes = test_cfg.get("beam_size", 0)
        beam_alpha = test_cfg.get("alpha", 0)
    else:
        beam_sizes = 0
        beam_alpha = 0
    if isinstance(beam_sizes, int):
        beam_sizes = [beam_sizes]
    assert beam_alpha >= 0, "Use alpha >= 0"

    for beam_size in beam_sizes:
        for data_set_name, data_set in data_to_predict.items():

            #pylint: disable=unused-variable
            scores, sources, sources_raw, references, hypotheses, \
            hypotheses_raw, attention_scores, scores_by_lang, by_lang = validate_on_data(
                model, data=data_set, batch_size=batch_size,
                batch_type=batch_type,
                src_level=src_level, trg_level=trg_level,
                max_output_length=max_output_length, eval_metrics=eval_metric,
                attn_metrics=attn_metric,
                use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
                beam_alpha=beam_alpha, save_attention=save_attention)
            #pylint: enable=unused-variable

            if "trg" in data_set.fields:
                labeled_scores = sorted(scores.items())
                eval_report = ", ".join("{}: {:.5f}".format(n, v)
                                        for n, v in labeled_scores)
                decoding_description = "Greedy decoding" if beam_size == 0 else \
                    "Beam search decoding with beam size = {} and alpha = {}".\
                        format(beam_size, beam_alpha)
                logger.info("%4s %s: [%s]",
                            data_set_name, eval_report, decoding_description)
                if scores_by_lang is not None:
                    for metric, scores in scores_by_lang.items():
                        # make a report
                        lang_report = [metric]
                        numbers = sorted(scores.items())
                        lang_report.extend(["{}: {:.5f}".format(k, v)
                                            for k, v in numbers])

                        logger.info("\n\t".join(lang_report))
            else:
                logger.info("No references given for %s -> no evaluation.",
                            data_set_name)

            if save_attention:
                # currently this will break for transformers
                if attention_scores:
                    #attention_name = "{}.{}.att".format(data_set_name, step)
                    #attention_path = os.path.join(model_dir, attention_name)
                    logger.info("Saving attention plots. This might take a while..")
                    store_attention_plots(attentions=attention_scores,
                                          targets=hypotheses_raw,
                                          sources=[s for s in data_set.src],
                                          indices=range(len(hypotheses)),
                                          model_dir=model_dir,
                                          steps=step,
                                          data_set_name=data_set_name)
                    logger.info("Attention plots saved to: %s", model_dir)
                else:
                    logger.warning("Attention scores could not be saved. "
                                   "Note that attention scores are not available "
                                   "when using beam search. "
                                   "Set beam_size to 0 for greedy decoding.")

            if output_path is not None:
                for lang, ref_and_hyp in by_lang.items():
                    if lang is None:
                        # monolingual case
                        output_path_set = "{}.{}".format(output_path, data_set_name)
                    else:
                        output_path_set = "{}.{}.{}".format(output_path, lang, data_set_name)
                    if isinstance(ref_and_hyp[0], str):
                        hyps = ref_and_hyp
                    else:
                        hyps = [hyp for (ref, hyp) in ref_and_hyp]
                    with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                        for hyp in hyps:
                            out_file.write(hyp + "\n")
                    logger.info("Translations saved to: %s", output_path_set)


def translate(cfg_file, ckpt: str, output_path: str = None) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    """

    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name+tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix, field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        _, _, _, _, hypotheses, _, _, _, _ = validate_on_data(
            model, data=test_data, batch_size=batch_size, level=level,
            max_output_length=max_output_length, eval_metrics=[],
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha)
        return hypotheses

    cfg = load_config(cfg_file)

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)

    data_cfg = cfg["data"]

    batch_size = cfg["training"].get("batch_size", 1)
    use_cuda = cfg["training"].get("use_cuda", False)
    max_output_length = cfg["training"].get("max_output_length", None)

    # read vocabs

    # This will need to change: currently translate does not support inflection
    src_vocab_file = data_cfg.get(
        "src_vocab", cfg["training"]["model_dir"] + "/src_vocab.txt")
    trg_vocab_file = data_cfg.get(
        "trg_vocab", cfg["training"]["model_dir"] + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)
    vocabs = {"src": src_vocab, "trg": trg_vocab}

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = list if level == "char" else str.split

    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], vocabs=vocabs)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", 0)
    else:
        beam_size = 0
        beam_alpha = 0
    if beam_alpha < 0:
        raise ConfigurationError("alpha for length penalty should be >= 0")

    if not sys.stdin.isatty():
        # file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            print("Translations saved to: {}".format(output_path_set))
        else:
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
