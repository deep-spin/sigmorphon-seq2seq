# Simple class for learning an alignment of strings, MED-style.
# Weights are learned by a Chinese Restaurant Process sampler
# that weights single alignments x:y in proportion to how many times
# such an alignment has been seen elsewhere out of all possible alignments.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Usage:
# Align(wordpairs) <= wordpairs is an iterable of 2-tuples
# The resulting Align.alignedpairs is a list of aligned 2-tuples

# Relies on C-code in libalign.so built from align.c through ctypes.
# Author: Mans Hulden
# MH20151102
# Some alterations by Ben Peters

from itertools import count
from ctypes import c_int, c_void_p, cdll, POINTER

libalign = cdll.LoadLibrary('./libalign.so')

libalign_add_int_pair = libalign.add_int_pair
libalign_clear_counts = libalign.clear_counts
libalign_initial_align = libalign.initial_align
libalign_crp_train = libalign.crp_train
libalign_crp_align = libalign.crp_align
libalign_med_align = libalign.med_align

libalign_getpairs_init = libalign.getpairs_init
libalign_getpairs_init.restype = c_void_p
libalign_getpairs_in = libalign.getpairs_in
libalign_getpairs_in.restype = POINTER(c_int)
libalign_getpairs_out = libalign.getpairs_out
libalign_getpairs_out.restype = POINTER(c_int)
libalign_getpairs_advance = libalign.getpairs_advance
libalign_getpairs_advance.restype = c_void_p
libalign_align_init = libalign.align_init
libalign_align_init.restype = None
# libalign_align_init_with_seed = libalign.align_init_with_seed
libalign_align_init.restype = None


class Aligner:

    def __init__(self, wordpairs, align_symbol=u' ', iterations=10,
                 burnin=5, lag=1, mode='crp', random_seed=None):
        s = set(u''.join((x[0] + x[1] for x in wordpairs)))
        self.s2i = dict(zip(s, range(1, len(s) + 1)))
        self.i2s = {v: k for k, v in self.s2i.items()}
        self.i2s[0] = align_symbol
        # Map stringpairs to -1 terminated integer sequences
        intpairs = []
        for src, trg in wordpairs:
            intin = [self.s2i[s] for s in src] + [-1]
            intout = [self.s2i[t] for t in trg] + [-1]
            intpairs.append((intin, intout))

        '''
        if random_seed:
            libalign_align_init_with_seed(random_seed)
        else:
            libalign_align_init()
        '''
        libalign_align_init()

        for i, o in intpairs:
            icint = (c_int * len(i))(*i)
            ocint = (c_int * len(o))(*o)
            libalign_add_int_pair(icint, ocint)

        # Run CRP align
        libalign_clear_counts()
        libalign_initial_align()
        if mode == 'crp':
            libalign_crp_train(c_int(iterations), c_int(burnin), c_int(lag))
            libalign_crp_align()
        else:
            libalign_med_align()

        # Reconvert to output
        self.alignedpairs = []
        stringpairptr = libalign_getpairs_init()
        while stringpairptr is not None:
            inints = libalign_getpairs_in(c_void_p(stringpairptr))
            instr = []
            for j in count():
                if inints[j] == -1:
                    break
                instr.append(self.i2s[inints[j]])

            outints = libalign_getpairs_out(c_void_p(stringpairptr))
            outstr = []
            for j in count():
                if outints[j] == -1:
                    break
                outstr.append(self.i2s[outints[j]])

            self.alignedpairs.append((''.join(instr), ''.join(outstr)))
            stringpairptr = libalign_getpairs_advance(c_void_p(stringpairptr))
