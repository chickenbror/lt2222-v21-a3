"""
To run in CLI:
python3 eval.py MODEL.pt TEST.txt DESTINATION.txt

Creates DESTINATIONS.txt with replace/predicted vowels
Also prints on screen the accuracy of MODEL
"""


import sys
import argparse
import numpy as np
import pandas as pd
from model import train
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

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


def replace_vowels(tokens, predictions):
    text = []
    i=0
    for t in tokens:
        if t not in ['<s>','<e>']:
            if t in vowels:
                idx = predictions[i] # start from the front of the predictions list and loop through
                i+=1
                pred_vowel = vowels[idx]  # change idx back to vowel
                text.append(pred_vowel)
            else:
                text.append(t)

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("test_file", type=str)
    parser.add_argument("destination", type=str)
    
    args = parser.parse_args()

    # The trained model
    model = torch.load(args.model_file)
    model.eval()
    train_vocab = model.vocab

    # Process the test data text
    test_tokens, test_vocab = process_text(args.test_file)

    # Loop through text data's tokens too, 
    # but the one-hot vectors use vocab-list from train data instead (so the vocab sizes are consistent) 
    test_vowel_idxes, test_vowel_feats = find_vowels(test_tokens, train_vocab)
    
    with torch.no_grad():
        output = model( torch.Tensor(test_vowel_feats) ) # Apply trained model on the context-features from test data
        predictions = torch.max(output.data, dim=1)[1].tolist() # list of predicted vowel idxes, based on neighbouring features

    # Write replaced vowels to a new text file
    with open(args.destination, 'w') as f:
        text = ''.join( replace_vowels(test_tokens, predictions) )
        f.write(text)

    # Print accuracy
    golds = test_vowel_idxes
    total_corrects = sum([1 if pred==gold else 0 for (pred,gold) in zip(predictions, golds)])
    accuracy = total_corrects / len(golds)
    print(f'Accuracy of {args.model_file}: {100*accuracy} %')
