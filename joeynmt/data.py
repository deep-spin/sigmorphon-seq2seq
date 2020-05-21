# coding: utf-8
"""
Data module
"""
import sys
from os.path import isfile
from functools import partial

import torch

from torchtext.datasets import TranslationDataset
from torchtext.data import Dataset, Iterator, Field, BucketIterator

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab
from joeynmt.datasets import MonoDataset, SigmorphonDataset, \
    SimpleSigmorphonDataset, SigmorphonG2PDataset


def filter_example(ex, max_sent_length):
    return len(ex.src) <= max_sent_length and len(ex.trg) <= max_sent_length


def load_data(data_cfg: dict):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - vocabs: dictionary from src and trg (and possibly other fields) to
            their corresponding vocab objects
    """
    data_format = data_cfg.get("format", "bitext")
    formats = {"bitext", "sigmorphon", "sigmorphon-simple", "sigmorphon-g2p"}
    assert data_format in formats

    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)

    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]
    default_level = data_cfg.get("level", "word")
    voc_limit = data_cfg.get("voc_limit", sys.maxsize)
    voc_min_freq = data_cfg.get("voc_min_freq", 1)

    main_fields = ["src", "trg"] if data_format != "sigmorphon" \
        else ["src", "trg", "inflection"]

    label_fields = []
    multilingual = data_cfg.get("multilingual", False)
    if multilingual:
        assert data_format in {"sigmorphon", "sigmorphon-g2p"}
        label_fields.append("language")

    suffixes = {f_name: data_cfg.get(f_name, "") for f_name in main_fields}

    sequential_field = partial(
        Field,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=lowercase,
        include_lengths=True
    )

    fields = dict()
    for f_name in main_fields:
        init_token = BOS_TOKEN if f_name == "trg" else None
        if f_name != "inflection":
            current_level = data_cfg.get(f_name + "_level", default_level)
            tok_fun = list if current_level == "char" else str.split
        else:
            tok_fun = partial(str.split, sep=';')
        fields[f_name] = sequential_field(
            init_token=init_token, tokenize=tok_fun)

    for f_name in label_fields:
        fields[f_name] = Field(sequential=False)

    filter_ex = partial(filter_example, max_sent_length=max_sent_length)

    if data_format == "bitext":
        dataset_cls = partial(
            TranslationDataset,
            exts=("." + suffixes["src"], "." + suffixes["trg"]),
            fields=(fields["src"], fields["trg"])
        )
    else:
        sigmorphon_fields = {k: [(k, v)] for k, v in fields.items()}
        if data_format == "sigmorphon":
            class_name = SigmorphonDataset
        elif data_format == "sigmorphon-g2p":
            class_name = SigmorphonG2PDataset
        else:
            class_name = SimpleSigmorphonDataset
        dataset_cls = partial(class_name, fields=sigmorphon_fields)

    if test_path is not None:
        trg_suffix = suffixes["trg"]
        if data_format != "bitext" or isfile(test_path + "." + trg_suffix):
            test_dataset_cls = dataset_cls
        else:
            test_dataset_cls = partial(
                MonoDataset, ext="." + suffixes["src"], field=fields["src"]
            )
    else:
        test_dataset_cls = None

    train_data = dataset_cls(path=train_path, filter_pred=filter_ex)

    vocabs = dict()
    vocab_counts = dict()  # language-specific vocab subsets

    for f_name in main_fields:
        vocab_file = data_cfg.get("{}_vocab".format(f_name), None)
        max_size = data_cfg.get("{}_voc_limit".format(f_name), voc_limit)
        min_freq = data_cfg.get("{}_voc_min_freq".format(f_name), voc_min_freq)
        f_vocab, f_vocab_counts = build_vocab(
            field=f_name,
            min_freq=min_freq,
            max_size=max_size,
            dataset=train_data,
            vocab_file=vocab_file,
            multilingual=multilingual)
        vocabs[f_name] = f_vocab
        vocab_counts[f_name] = f_vocab_counts

    for f_name in label_fields:
        vocab_file = data_cfg.get("{}_vocab".format(f_name), None)
        max_size = data_cfg.get("{}_voc_limit".format(f_name), voc_limit)
        min_freq = data_cfg.get("{}_voc_min_freq".format(f_name), voc_min_freq)
        f_vocab, _ = build_vocab(
            field=f_name,
            min_freq=min_freq,
            max_size=max_size,
            dataset=train_data,
            vocab_file=vocab_file,
            sequential=False)
        vocabs[f_name] = f_vocab

    dev_data = dataset_cls(path=dev_path)

    if test_path is not None:
        trg_suffix = suffixes["trg"]
        if data_format != "bitext" or isfile(test_path + "." + trg_suffix):
            test_dataset_cls = dataset_cls
        else:
            test_dataset_cls = partial(
                MonoDataset, ext="." + suffixes["src"], field=fields["src"]
            )
        test_data = test_dataset_cls(path=test_path)
    else:
        test_data = None

    for field_name in fields:
        fields[field_name].vocab = vocabs[field_name]

    ret = {"train_data": train_data, "dev_data": dev_data,
           "test_data": test_data, "vocabs": vocabs}

    return ret


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    data_iter = BucketIterator(
        repeat=False,
        sort=False,
        dataset=dataset,
        batch_size=batch_size,
        batch_size_fn=batch_size_fn,
        train=train,
        sort_within_batch=train,
        shuffle=shuffle if train else False,
        sort_key=lambda x: len(x.src) if train else None)

    return data_iter
