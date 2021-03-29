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

# Adpated from a()
def process_text(filename):
    tokens = []
    with open(filename, "r") as text:
        for l in text:  #for each line in the opened file
            tokens += [c for c in l] #extend each line's character-tokens to mm 

    tokens = ["<s>", "<s>"] + tokens + ["<e>", "<e>"]  # add start/end tokens
    return tokens, list(set(tokens)) #two values: mm  with or without repeated char-tokens

# Adpated from g()
def onehot_vectorize(token, vocab): 
    vec_array = np.zeros(len(vocab)) # array of 0; len of array==nr of unique tokens
    if token in vocab:
        vec_array[vocab.index(token)] = 1  # change the nth 0 to 1
    return vec_array # np array representing a tokens's one-hot vector

# Adpated from b()
def find_vowels(tokens, vocab):
    vowel_idxes = []
    vowel_features = []
    for i in range(len(tokens) - 4): #indexing 0 ~ -5 => ie, after adding 2 >> from first token to last token, excl the <s>/<e>
        if tokens[i+2] not in vowels: # if the char-token (indexed 2 ~ -3) isn't a vowel
            continue             # don't run the next 4 lines, and instead loop the next char-token
        
        #(Below only applies if char-token is a vowel)
        vowel_idx = vowels.index(tokens[i+2]) # encode the char-token to vowels-list idx
        vowel_idxes.append(vowel_idx)  # gt=list of vowel idxes
        feat = np.concatenate([onehot_vectorize(token, vocab) for token in [tokens[i], tokens[i+1], tokens[i+3], tokens[i+4]]]) # concatenated array of one-hot vectors of 4 neighbouring tokens (2 before, 2 after)
        vowel_features.append(feat) # list of context-features (ie, concatenation of 4 neighbouring one-hot vectors)

    return np.array(vowel_idxes), np.array(vowel_features) # list of indexed vowels, list of concatenated one-hot vectors

def replace_vowels(tokens, predictions, destination):
    text = []
    i=0
    for t in tokens:
        if t not in ['<s>','<e>']:
            if t in vowels:
                idx = predictions[i] # start from the front of the predictions list and loop over
                i+=1
                pred_vowel = vowels[idx]  # change idx back to vowel
                text.append(pred_vowel)
            else:
                text.append(t)

    # Write replaced vowels to a new text file
    with open(destination, 'w') as f:
        joined_text = ''.join( text )
        f.write(joined_text)


def train_models(k,r):
    m='svtrain.lower.txt' #train data
    h=f'model_k{k}_r{r}.pt' # model filename


    #Train and save model
    q = a(m) # tokenise => tokens list, vocab list
    w = b(q[0], q[1]) # find vowels from tokens list => list of indexed vowels, list of each vowel's context-features
    t = train(w[0], w[1], q[1], k, r) # vowels, features, vocab, hiddensize, epochs
    torch.save(t, h) # save trained model to destination


def eval_models(k,r):
    m='svtrain.lower.txt' #train data
    h=f'model_k{k}_r{r}.pt' # model filename

    # The trained model
    model = torch.load(h)
    model.eval()
    train_vocab = model.vocab

    # Process the test data text
    test_tokens, test_vocab = process_text('svtest.lower.txt')

    # Loop through text data's tokens too, 
    # but the one-hot vectors use vocab-list from train data instead (so the vocab sizes are consistent) 
    test_vowel_idxes, test_vowel_feats = find_vowels(test_tokens, train_vocab)
    
    with torch.no_grad():
        output = model( torch.Tensor(test_vowel_feats) ) # Apply trained model on the context-features from test data
        predictions = torch.max(output.data, dim=1)[1].tolist() # list of predicted vowel idxes, based on neighbouring features

    # Print accuracy
    golds = test_vowel_idxes
    total_corrects = sum([1 if pred==gold else 0 for (pred,gold) in zip(predictions, golds)])
    accuracy = total_corrects / len(golds)
    print(f'HiddenSize (k)= {k}; Epochs (r)= {r}')
    print(f'Accuracy: {100*accuracy} %\n')

    replace_vowels(test_tokens, predictions, f'text_k{k}_r{r}')

    

if __name__ == "__main__":
    
    ks=[50, 100, 200, 400, 800] # hiddensize, default 200
    rs=[50, 100, 200, 400, 800] # epochs, default 100

    # #TRAINING MODELS
    # # change k and keep r constant
    # for k in ks:
    #     r=100
    #     train_models(k,r)

    # # change r and keep k constant
    # for r in rs:
    #     k=200
    #     train_models(k,r)


    #EVALUATION
    # change k and keep r constant
    for k in ks:
        r=100
        eval_models(k,r)

    # change r and keep k constant
    for r in rs:
        k=200
        eval_models(k,r)

