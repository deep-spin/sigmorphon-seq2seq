#!/usr/bin/env python

import argparse
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from plotnine import ggplot, aes, geom_point


def read_embeddings(path):
    labels, vectors = [], []
    with open(path) as f:
        for line in f:
            toks = line.strip().split('\t')
            labels.append(toks[0])
            vectors.append([float(t) for t in toks[1:]])

    return labels, np.array(vectors, dtype=np.float)


def build_tsne(path, **kwargs):
    labels, vectors = read_embeddings(path)
    tsne_vec = TSNE(**kwargs).fit_transform(vectors)
    df = pd.DataFrame(tsne_vec, columns=["x", "y"])
    df["label"] = labels
    return df


def main(opts):
    labels, vectors = read_embeddings(opt.embeddings)
    tsne_vec = TSNE().fit_transform(vectors)
    print(tsne_vec.shape)
    df = pd.DataFrame(tsne_vec, columns=["x", "y"])
    df["label"] = labels
    print(df)
    ggplot(aes(x="x", y="y", color="label"), data=df) + geom_point()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    opt = parser.parse_args()
    main(opt)
