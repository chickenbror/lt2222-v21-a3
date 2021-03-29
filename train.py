import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["USE_CPU"]="1"

import sys
import argparse
import numpy as np
import pandas as pd
from model import train
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def a(f):
    mm = []
    with open(f, "r") as q:
        for l in q:  #for each line in the opened file
            mm += [c for c in l] #extend each line's character-tokens to mm 

    mm = ["<s>", "<s>"] + mm + ["<e>", "<e>"]  # add start/end tokens
    return mm, list(set(mm)) #two values: mm  with or without repeated char-tokens

def g(x, p): # single token & list of unique tokens
    z = np.zeros(len(p)) # array of 0; len of array==nr of unique tokens
    z[p.index(x)] = 1  # change the nth 0 that has the same index of x in p
    return z # np array representing a x's one-hot vector

def b(u, p): #char-tokens list; unique char-tokens list
    gt = []
    gr = []
    for v in range(len(u) - 4): #indexing 0 ~ -5 => ie, after adding 2 >> from first token to last token, excl the <s>/<e>
        if u[v+2] not in vowels: # if the char-token (indexed 2 ~ -3) isn't a vowel
            continue             # don't run the next 4 lines, and instead loop the next char-token
        
        #(Below only applies if char-token is a vowel)
        h2 = vowels.index(u[v+2]) # encode the char-token to vowels-list idx
        gt.append(h2)  # gt=list of vowel idxes
        r = np.concatenate([g(x, p) for x in [u[v], u[v+1], u[v+3], u[v+4]]]) # concatenated array of one-hot vectors of 4 neighbouring tokens (2 before, 2 after)
        gr.append(r) # list of context-features (ie, concatenation of 4 neighbouring one-hot vectors)

    return np.array(gr), np.array(gt) # list of indexed vowels, list of concatenated one-hot vectors
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", dest="k", type=int, default=200)
    parser.add_argument("--r", dest="r", type=int, default=100)
    parser.add_argument("m", type=str)
    parser.add_argument("h", type=str)
    # add_argument(args...)
    
    args = parser.parse_args()

    q = a(args.m) # tokenise => tokens list, vocab list
    w = b(q[0], q[1]) # find vowels from tokens list => list of indexed vowels, list of each vowel's context-features
    t = train(w[0], w[1], q[1], args.k, args.r) # vowels, features, vocab, hiddensize, epochs

    torch.save(t, args.h) # save trained model to destination
