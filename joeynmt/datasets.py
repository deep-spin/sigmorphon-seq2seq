# coding: utf-8

from os.path import expanduser, basename, splitext
from glob import glob
import re
from itertools import chain
import unicodedata

from torchtext.data import Dataset, Field, Example


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(Example.fromlist([src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)


def lang_name(path, year=2020):
    filename = basename(path)
    if year == 2020:
        return splitext(filename)[0]
    else:
        filename = re.sub(r"-(un)?covered", "", filename)
        match = re.match(r'.+(?=-(train|dev|test|covered|uncovered))',
                         filename)
        return match.group(0) if match else None


def g2p_lang_name(path):
    return lang_name(path).split("_")[0]


def data_setting(path):
    filename = basename(path)
    return filename.split("-")[-1]


def has_trg(path):
    with open(path) as f:
        return len(f.readline().strip().split("\t")) == 3


class SigmorphonDataset(Dataset):
    """Adapted from NNowledge"""

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "trg"):
            return len(ex.src), len(ex.trg)
        return len(ex.src)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(SigmorphonDataset, self).__reduce_ex__()

    def __init__(self, fields, path, filter_pred=None):

        paths = glob(path) if isinstance(path, str) else path
        assert len(paths) > 0
        paths.sort()
        examples = []
        for p in paths:
            with open(p) as f:
                language = lang_name(p) if 'language' in fields else None

                for line in f:
                    line = line.strip()
                    if line:
                        ex_dict = dict()
                        if language is not None:
                            ex_dict["language"] = language
                        line_fields = line.strip().split('\t')
                        if len(line_fields) == 3:
                            src, trg, inflection = line_fields
                            ex_dict['trg'] = trg
                        else:
                            src, inflection = line_fields
                            fields.pop("trg", None)  # hmm

                        ex_dict["src"] = src
                        ex_dict["inflection"] = inflection

                        ex = Example.fromdict(ex_dict, fields)
                        examples.append(ex)

        fields = dict(chain.from_iterable(fields.values()))
        super(SigmorphonDataset, self).__init__(examples, fields, filter_pred)


class SigmorphonG2PDataset(Dataset):

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "trg"):
            return len(ex.src), len(ex.trg)
        return len(ex.src)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(SigmorphonG2PDataset, self).__reduce_ex__()

    def __init__(self, fields, path, filter_pred=None, decompose=True):

        paths = glob(path) if isinstance(path, str) else path
        assert len(paths) > 0
        paths.sort()
        examples = []
        for p in paths:
            with open(p) as f:
                language = g2p_lang_name(p) if 'language' in fields else None

                for line in f:
                    line = line.strip()
                    if line:
                        ex_dict = dict()
                        if language is not None:
                            ex_dict["language"] = language
                        line_fields = line.strip().split('\t')
                        assert 0 < len(line_fields) <= 2
                        src = line_fields[0]
                        if decompose:
                            # hard-coding the Korean decomposition
                            src = unicodedata.normalize("NFD", src)
                        ex_dict["src"] = src
                        if len(line_fields) == 2:
                            ex_dict['trg'] = line_fields[1]
                        else:
                            fields.pop("trg", None)

                        ex = Example.fromdict(ex_dict, fields)
                        examples.append(ex)

        fields = dict(chain.from_iterable(fields.values()))
        super(SigmorphonG2PDataset, self).__init__(
            examples, fields, filter_pred)


class SimpleSigmorphonDataset(Dataset):
    """
    Adapted from NNowledge. This dataset interprets the inflections as part of
    the src sequence. It requires src_level == word and trg_level == char, for
    bad reasons.
    """

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "trg"):
            return len(ex.src), len(ex.trg)
        return len(ex.src)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(SimpleSigmorphonDataset, self).__reduce_ex__()

    def __init__(self, fields, path, filter_pred=None, lang_src=False):
        if isinstance(path, str):
            path = [path]
        examples = []
        for p in path:
            with open(p) as f:
                language = lang_name(p) if 'language' in fields else None

                for line in f:
                    line = line.strip()
                    if line:
                        ex_dict = dict()
                        if language is not None:
                            ex_dict["language"] = language
                        line_fields = line.strip().split('\t')
                        if len(line_fields) == 3:
                            src, trg, inflection = line_fields
                            ex_dict['trg'] = trg
                        else:
                            src, inflection = line_fields
                            fields.pop("trg", None)  # hmm

                        # kludgey stuff for handling inflections
                        respaced_inflection = " ".join(inflection.split(";"))
                        respaced_src = " ".join(
                            [c if c != " " else "<space>" for c in src])
                        src_seq = []
                        if language is not None and lang_src:
                            src_seq.append(language)
                        src_seq.extend([respaced_inflection, respaced_src])

                        ex_dict["src"] = " ".join(src_seq)

                        ex = Example.fromdict(ex_dict, fields)
                        examples.append(ex)

        fields = dict(chain.from_iterable(fields.values()))
        super(SimpleSigmorphonDataset, self).__init__(
            examples, fields, filter_pred)
