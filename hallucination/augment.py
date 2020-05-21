"""
A script for generating hallucinated training data for the SIGMORPHON 2020
shared task.

The code is adapted from Anastasopoulos and Neubig's submission
to the SIGMORPHON 2019 shared task:
- code: https://github.com/antonisa/inflection
- paper: https://www.aclweb.org/anthology/D19-1091.pdf
"""

import align
import argparse
from os.path import join
from random import random, choice
import re
from itertools import chain, groupby


def read_data(filename):
    inputs = []
    outputs = []
    tags = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                src, trg, infl = line.split("\t")
                inputs.append(list(src.strip()))
                outputs.append(list(trg.strip()))
                tags.append(re.split(r'\W+', infl.strip()))

    return inputs, outputs, tags


def ends(group):
    ix = [i for i, v in group]
    return ix[0], ix[-1] + 1


def find_good_range(src, trg, min_len, max_len):
    mask = [s == t != " " for s, t in zip(src, trg)]
    if not any(mask):
        # Sometimes the alignment is off-by-one
        # This can also happen in cases of suppletion, i.e. go -> went
        trg = ' ' + trg
        mask = [s == t != " " for s, t in zip(src, trg)]
    mask_pos = enumerate(mask)
    ranges = [ends(g) for k, g in groupby(mask_pos, key=lambda x: x[1]) if k]
    ranges = [c for c in ranges if min_len <= c[1] - c[0] <= max_len]
    return ranges


def ngrams(seq, n):
    return list(zip(*[seq[i:] for i in range(n)]))


class Hallucinator(object):
    def __init__(self, *corpora, p=0.5, n=1):
        self.p = p  # probability of hallucinating a character
        words = list(chain.from_iterable(corpora))  # list of lists of chars

        assert n in {0, 1}
        self.chars = [c for c in chain.from_iterable(words) if c != " "]
        if n == 0:
            self.chars = list(set(self.chars))

    def sample(self, gold_seq):
        return [choice(self.chars) if random() <= self.p else g
                for g in gold_seq]


def augment(inputs, outputs, tags, hallucinator, min_len=3, max_len=10):
    temp = [(''.join(inp), ''.join(out)) for inp, out in zip(inputs, outputs)]
    # aligned returns pairs of strings with spaces for null alignments
    aligned = align.Aligner(temp).alignedpairs

    new_inputs = []
    new_outputs = []
    new_tags = []
    for k, (src, trg) in enumerate(aligned):
        good_ranges = find_good_range(src, trg, min_len, max_len)
        if good_ranges:
            new_src, new_trg = list(src), list(trg)
            for good_range in good_ranges:
                s, e = good_range
                gold_seq = new_src[s: e]
                hallucinated_seq = hallucinator.sample(gold_seq)
                new_src[s: e] = new_trg[s: e] = hallucinated_seq

            # trim, unless src and trg have an aligned whitespace
            new_i1 = [c for i, c in enumerate(new_src)
                      if (c.strip() or (new_src[i] == new_trg[i] == ' '))]
            new_o1 = [c for i, c in enumerate(new_trg)
                      if (c.strip() or (new_src[i] == new_trg[i] == ' '))]

            new_inputs.append(new_i1)
            new_outputs.append(new_o1)
            new_tags.append(tags[k])

    return new_inputs, new_outputs, new_tags


def get_chars(words):
    return [char for char in chain.from_iterable(words) if char != " "]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=str, help="path to data")
    parser.add_argument("language", type=str, help="language")
    parser.add_argument("--examples", default=10000, type=int,
                        help="number of examples to hallucinate (def: 10000)")
    parser.add_argument("--use_dev", action="store_true",
                        help="whether to use the development set (def: False)")
    parser.add_argument("--n", default=1, type=int,
                        help="degree of n-gram model to hallucinate from")
    parser.add_argument("--p", default=0.5, type=float,
                        help="probability of hallucinating at each time step")
    args = parser.parse_args()

    data_path = args.datapath
    language = args.language
    train_path = join(data_path, language + ".trn")

    N = args.examples
    use_dev = args.use_dev

    true_src, true_trg, true_inf = read_data(train_path)

    if use_dev:
        dev_path = join(data_path, language + ".dev")
        dev_src, dev_trg, dev_inflection = read_data(dev_path)
        true_src.extend(dev_src)
        true_trg.extend(dev_trg)
        true_inf.extend(dev_inflection)

    hal = Hallucinator(true_src, true_trg, n=args.n)

    src, trg, inflection = [], [], []
    while len(src) < N:
        h_src, h_trg, h_inf = augment(true_src, true_trg, true_inf, hal)

        src.extend(h_src)
        trg.extend(h_trg)
        inflection.extend(h_inf)
        if not h_src:
            break
    src = src[:N]
    trg = trg[:N]
    inflection = inflection[:N]

    with open(join(data_path, language + ".trn.hal"), 'w') as outp:
        for s, t, infl in zip(src, trg, inflection):
            ex = "\t".join([''.join(s), ''.join(t), ';'.join(infl)])
            outp.write(ex + '\n')


if __name__ == "__main__":
    main()
