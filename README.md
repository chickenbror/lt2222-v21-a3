
# LT2222 V21 Assignment 3

Your name: Hsien-hao Liao

## Part 1

#### a(f):
Reads a text file and returns a two-value tuple: a list of char-tokens & a list of non-repeating char-tokens (ie the vocab-list).

Arg: 

   f: string representing the filepath/filename.
    
Returns:

   The abovementioned tuple of two lists. List1 has doubled start/end \<s> and \<e> tokens plus all char-tokens, whereas the List2 has single start/end tokens plus non-repeating char-tokens (ie the vocab-list).

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

In CLI:

	python3 eval.py MODEL.pt TEST_DATA.txt DESTINATION.txt

All three functions from train.py were copied and pasted (with variable names changed for better readability) to eval.py to train to process the text and return a list of vowels' context-features as done in train.py. 
The difference is that the vocabulary sizes are not the same, so b() would take the vocab of train data as its parameter instead to give the one-shot vectors the same length. Also there are some char-tokens that exist only in either train or test data, so that a zeros array only becomes a one-hot vector on the condition that the given char-token exists in the train data vocab.


## Part 3

#### Varying k and holding r:
HiddenSize (k)= 50; Epochs (r)= 100
Accuracy: 21.767599046498294 %

HiddenSize (k)= 100; Epochs (r)= 100
Accuracy: 29.305090371443228 %

HiddenSize (k)= 200; Epochs (r)= 100
Accuracy: 39.627651549440266 %

HiddenSize (k)= 400; Epochs (r)= 100
Accuracy: 26.70408255635014 %

HiddenSize (k)= 800; Epochs (r)= 100
Accuracy: 9.666274403307082 %


#### Holding k and varying r:
HiddenSize (k)= 200; Epochs (r)= 50
Accuracy: 31.458012733472135 %

HiddenSize (k)= 200; Epochs (r)= 100
Accuracy: 39.627651549440266 %

HiddenSize (k)= 200; Epochs (r)= 200
Accuracy: 30.754956096677834 %

HiddenSize (k)= 200; Epochs (r)= 400
Accuracy: 11.772426903231645 %

HiddenSize (k)= 200; Epochs (r)= 800
Accuracy: 5.162789294227694 %

In both cases, the accuracy rates increased and then drop significantly low (with rendered the resulting texts jibberish-like). It appears that with increase epochs, the models get better at predicting the vowel from the neighboring characters, but also starts to overfit and thus making the wrong prediction at very high epochs. Also, similar to the observation from Assignment2, the more frequent vowels were better predicted (eg, 'é' only occurs 58 times in test text while 'a' occurs 16640 times). Common features patterns, ie, the context of common stop words, also were better predicted. 

## Bonuses

## Other notes
