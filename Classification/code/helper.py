from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np 
import time
import datetime

# ----------------------------- Contains all the helper functions ------------------------------- #

def calculate_dot_product(input_ls, weights):
        dot_product = 0
        for i,j in zip(input_ls, weights):
            dot_product += i*j
        return dot_product

def remove_punct(utterance):
    punctuation = ['.', ',', '?', '!', '\'', '\"']

    if isinstance(utterance, list):
        for element in punctuation:
            if element in utterance:
                utterance.remove(element)
    
    if isinstance(utterance, str):
        for element in punctuation:
            if element in utterance:
                utterance = utterance.strip(element)
    
    return utterance

def tokenizer(line):
    token_list = list()
    # Load stop words
    stop_words = stopwords.words('english')
    if '\n' in line:
        line = line.strip('\n')
    if '\"' in line:
        line = line.strip('\"')  
    tokens = list(map(lambda x:x.lower(), word_tokenize(line)))
    for word in tokens:
        if word not in stop_words:
            token_list.append(word)
    return token_list

def normalize(word_counts):
        
    probabilities = {} 
    denominator = 0 # normalizing factor
        
    for count in word_counts.values():
        denominator += count

    for word, count in word_counts.items():
        probabilities[word] = count / denominator 

    return probabilities

def utterance_to_vector(vocab, line):
    stop_words = stopwords.words('english')
    wordset = set()
    vector = []
    line = list(map(lambda x:x.lower(), word_tokenize(line)))
    for word in line:
            if word not in stop_words:
                wordset.add(word)
    for word in vocab:
        if word in wordset:
            vector.append(1)
        else:
            vector.append(0)
    return vector

def keywithmaxval(dictionary):
	max_val = max(zip(dictionary.values(), dictionary.keys()))[1]
	return max_val

def subtract_vector(v1, v2): 
    result = [x1 - x2 for (x1, x2) in zip(v1, v2)]
    return result

def add_vector(v1, v2): 
    result = [x1 + x2 for (x1, x2) in zip(v1, v2)]
    return result

def accuracy(vec1, vec2):
    num = 0
    for i in range(len(vec1)):
        if vec1[i] == vec2[i]:
            num += 1
    
    return num / len(vec1)

def concat_matrix(ls_1, ls_2):
    a = len(ls_1)
    b = len(ls_2)

    if a>b:
        c = a - b; sparse_arr = [0,0,0,0,0,0,0]
        for i in range(c):
            ls_2.append(sparse_arr)

        combined = np.column_stack((ls_1, ls_2))
        
    else:
        combined = np.column_stack((ls_1, ls_2[0:a]))

    return combined

def str_to_vec(vec_arr):
    arr = []
    for v in vec_arr:
        v = v.split('=')
        v = float(v[1])
        arr.append(v)
    return arr

def make_combined_vector(a,b):
    RES = list()
    if len(a)<len(b):
        for i,j in zip(a,b[0:len(a)]):
            res = [m+n for m,n in zip(i,j)]
            RES.append(res)
    else:
        for i,j in zip(a[0:len(b)],b):
            res = [m+n for m,n in zip(i,j)]
            RES.append(res)
    
    return RES

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))