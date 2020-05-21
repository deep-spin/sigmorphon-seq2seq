#!/usr/bin/env python

import argparse
import random
import sys


def multi_examples(dataset, size):
    while True:
        real_examples = [random.choice(dataset) for j in range(size)]
        srcs, trgs = zip(*real_examples)
        graphemes = " ".join(srcs)
        phonemes = " ".join(trgs)
        yield "\t".join([graphemes, phonemes])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("n", type=int)  # number of examples to make
    parser.add_argument("--size", type=int, default=2)  # words per new example
    opt = parser.parse_args()

    with open(opt.path) as f:
        g2p_data = [line.strip().split("\t") for line in f]

    examples = multi_examples(g2p_data, opt.size)
    for i in range(opt.n):
        sys.stdout.write(next(examples) + "\n")


if __name__ == "__main__":
    main()
