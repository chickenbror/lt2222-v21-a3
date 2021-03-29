# LT2222 V21 Assignment 3

Your name:

## Part 1

#### a(f):
Reads a text file and returns a two-value tuple: a list of char-tokens & a list of non-repeating char-tokens (ie the vocab-list).
Arg: 
    f: string representing the filepath/filename.
Returns:
    The abovementioned tuple of two lists. List1 has doubled start/end '<s>' and '<e>' tokens plus all char-tokens, whereas the List2 has single start/end tokens plus non-repeating char-tokens (ie the vocab-list).

#### g(x,p):
Takes a token and a vocab-list, and returns a numpy array that represents a one-hot vector.
Args: 
    x: a char-token
    p: a vocab-list which includes x
Returns: 
    The one-hot vector of x (in the form of a numpy array).

#### b(u,p): 
Loops over every char-token returned by a() and looks for vowels. Returns a list of found vowel-indexes and a list of the former's corresponding context-features.
Args: 
    u: a list of char-tokens which function a(textfile) returned
    p: a list of non-repeating char-tokens which function a(textfile) returned (ie, the vocab of textfile, in the sense that each vocab is a char )
Returns: 
    Two lists of the same length. List1 has all the vowels found in u, each of which is mapped to the corresponding index in 'vowels' list. List2 is a list of numpy arrays, each of which correspond to a vowel from List1, and is the concatenation of 4 one-hot vectors (of the 2 tokens before/after the vowel-token).

#### CLI args:
    m: the text file path
    h: destination to save the trained model
    --k: hiddensize (parameter of train(); 200 by default if left blank )
    --r: epochs (parameter of train(); 100 by default if left blank )


## Part 2

## Part 3

## Bonuses

## Other notes
