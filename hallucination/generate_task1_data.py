#!/usr/bin/env python

import argparse
from itertools import groupby, count, chain
from collections import Counter

import align


def read_task1_data(path):
    with open(path) as f:
        src, trg = zip(*[line.strip().split("\t") for line in f])
        #print(max(len(t) for ))
        # trg = [re.sub(r" ", "  ", t) for t in trg]
        return src, trg


def rules(pair):
    s, t = pair
    assert len(s) == len(t)
    # now: the source will have some consecutive non-whitespace indices
    # those same indices (actually the same indices? not necessarily)
    print(s)
    print(t)
    print()
    '''
    for i in range(0, len(s), 2):
        print(s[i:i+2], t[i:i+2])
    print()
    '''
    mappings = []
    # what are the indices of the src chunks?
    src_chunks = groupby(enumerate(s), key=lambda x: x[1] != " ")

    phoneme_chunks = (t for t in t.split())

    for value, chunk in src_chunks:
        if value:
            chunk = list(chunk)
            ix, letters = zip(*chunk)
            letters = "".join(letters)

            # I think this might not be the best way. Instead of using indices,
            # take the next
            '''
            if len(ix) == 1:
                phonemes = t[ix[0]]
            else:
                phonemes = t[ix[0]:ix[-1]]
            '''
            phonemes = next(phoneme_chunks)
            mappings.append((letters, phonemes))

    return mappings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    opt = parser.parse_args()

    src, trg = read_task1_data(opt.data)
    trg_inventory = set(chain.from_iterable([t.split() for t in trg]))
    fake_phonemes = (chr(i + 65) for i in range(26))
    real2fake = {p: next(fake_phonemes) for p in trg_inventory if len(p) > 1}
    fake2real = {v: k for k, v in real2fake.items()}
    trg = [" ".join([real2fake.get(c, c) for c in t.split()]) for t in trg]

    # corpus = zip(*zip(src, trg))
    corpus = src, trg
    corpus = list(zip(*corpus))
    pairs = align.Aligner(corpus, mode="crp", iterations=10).alignedpairs
    rule_counts = Counter(chain.from_iterable([rules(pair) for pair in pairs]))
    print(rule_counts.most_common(100))


if __name__ == "__main__":
    main()
