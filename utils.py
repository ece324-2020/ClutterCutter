import torch
import torch.nn as nn
import numpy as np

import torchtext
from torchtext import data
import spacy

import os
import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import string
import re 

import seaborn as sn
from sklearn.metrics import confusion_matrix

from googletrans import Translator

import os
from zipfile import ZipFile
from itertools import chain
from glob import glob
import re

#import os
import random
#import csv

import nltk
from nltk.corpus import wordnet

def reformat_txt(path):
    """
    Combines text files for each data label under one tsv file.
    """
    for dir in os.listdir(path):
        # remove previous copy
        if dir[-3:] == "tsv":
            os.remove(os.path.join(path, dir))

        elif dir != ".DS_Store":
            text = []
            tsv_path = os.path.join(path, dir, "text.tsv")

            # remove previous copy of tsv
            if os.path.exists(tsv_path):
                os.remove(tsv_path)

            # Extract text from each file
            for file in os.listdir(os.path.join(path, dir)):
                if file[-4:] == ".txt":
                    file_path = os.path.join(path, dir, file)
                    with open(file_path, 'r', encoding='utf8') as f:
                        phrase = f.read()
                        for n in phrase:
                          if n.isdigit():
                            phrase = phrase.replace(n, "")
                           
                        phrase = re.sub(r"won\'t", "will not", phrase)
                        phrase = re.sub(r"can\'t", "can not", phrase)
                        phrase = re.sub(r"n\'t", " not", phrase)
                        phrase = re.sub(r"\'re", " are", phrase)
                        phrase = re.sub(r"\'s", " is", phrase)
                        phrase = re.sub(r"\'d", " would", phrase)
                        phrase = re.sub(r"\'ll", " will", phrase)
                        phrase = re.sub(r"\'t", " not", phrase)
                        phrase = re.sub(r"\'ve", " have", phrase)
                        phrase = re.sub(r"\'m", " am", phrase)
                        phrase = re.sub(r'[^\w\s]', ' ', phrase)
                        phrase = re.sub(' +', ' ', phrase)
                        phrase = phrase.lower()
                        phrase.strip('\n')
                        phrase.strip('\t') # removes tabs so content can be stored in tsv format.
                        phrase = ' '.join([w for w in phrase.split() if len(w) > 2]) # remove words with len <= 2
                        text.append([phrase])

        # Merge all text data into one tsv        
        with open(tsv_path, 'w', encoding='utf8') as f:
            f.write("text\n")
            writer = csv.writer(f)
            writer.writerows(text)      

def load_df(path):
    """
    Creates a Pandas Dataframe from the text.tsv file containing text data.

    Returns Dataframe object with the proper labels assigned to each piece of text.
    """
    labels = {'Academics': 0, 'Alerts': 1, 
              'Personal': 2, 'Professional': 3,
              'Promotions and Events': 4}
    columns = ['text', 'label']
    df = pd.DataFrame(columns=columns)
    for dir in os.listdir(path):
        if dir != ".DS_Store":
            label = labels[dir]
            data_path = os.path.join(path, dir, "text.tsv")
            text_df = pd.read_csv(data_path, sep='\t')
            text_df.insert(1, 'label', label) # Adds label based on directory name
            df = pd.concat([df, text_df])
    df = df.reset_index(drop=True)
    return df

def pre_processing(path):
    data = load_df(path) # Load labelled dataframe 

    x_tot = data["text"]
    y_tot = data ["label"]

    #Splitting data into train (0.64), test (0.20), validation (0.16)
    x, x_test, y, y_test = train_test_split(x_tot,y_tot,test_size=0.2, train_size=0.8, random_state = 0, stratify = y_tot) # 0.8 = train + validation)
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2, random_state = 0, stratify = y) # 0.2 x 0.8 = 0.16 validation

    print("Total examples in test", x_test.shape)
    print("Total examples in train", x_train.shape)
    print("Total examples in validation", x_val.shape)

    print("\nTEST: Examples in each class\n", y_test.value_counts())
    print("\nTRAIN: Examples in each class\n", y_train.value_counts())
    print("\nVALIDATION: Examples in each class\n", y_val.value_counts())

    #Concatenate text and labels & load into tsv files 
    testdata = pd.concat([x_test, y_test], axis = 1)
    testdata.to_csv(os.path.join(path, "test.tsv"), sep="\t",index=False)

    traindata = pd.concat([x_train, y_train], axis = 1)
    traindata.to_csv(os.path.join(path, "train.tsv"), sep="\t",index=False)

    valdata = pd.concat([x_val, y_val], axis = 1)
    valdata.to_csv(os.path.join(path, "validation.tsv"), sep="\t",index=False)

    #Create 4th dataset (overfit) for debugging (pulls 10 samples per label to create a 50 sample dataset)
    overfit = traindata.groupby('label', group_keys=False).apply(lambda x: x.sample(10))
    overfit.to_csv(os.path.join(path, "overfit.tsv"), sep="\t",index=False)

def make_iter(path, batch_size):
    """
    Creates iterators and makes a vocab object.
    """
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
            path=path, train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
    sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    
    TEXT.build_vocab(train_data, val_data, test_data)
    TEXT.build_vocab(train_data, val_data, test_data, vectors='fasttext.simple.300d')

    # TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100)) # Get rid of this line for FastText 

    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape) 
    return train_iter, val_iter, test_iter, vocab

# Accuracy function (for both models)
def accuracy(predictions, labels):
    """
    Called in training loop (on training data), evaluate function (on evaluation data)
    Called after training loop on evaluate function (on test data)
    Returns decimal accuracy (0.0 - 1.0)
    """
    correct = 0 
    total = 0
    
    _, predicted = torch.max(predictions,1) #finds max, returns index of max (predicted label)
    total = total + len(labels) #can probably change to just len(labels) later
    correct = correct + (predicted.float() == labels).sum() 

    batchacc = correct.item() / total
    return batchacc

def evaluate(model, data_iter):
    """
    Called in training loop (on evaluation data)
    Called after training loop (on test data) 
    Returns decimal accuracy (0.0 - 1.0) and loss 
    
    """
    loss_fnc = nn.CrossEntropyLoss()
    correct = 0    
    
    batchloss_accum = 0.0
    batchacc_accum = 0.0
    
    for i,batch in enumerate(data_iter): # Go through each batch in data_iter
        batch_input, batch_input_length = batch.text
        outputs = model(batch_input, batch_input_length)
        
        #Calculate accuracy
        acc = accuracy(outputs, batch.label)
        batchacc_accum = batchacc_accum + acc
        
        #Calculate loss
        batchloss = loss_fnc(outputs, batch.label) 
        batchloss_accum = batchloss_accum + batchloss.item()

    avgbatchloss = batchloss_accum/len(data_iter)
    avgbatchacc = batchacc_accum/len(data_iter)
    
    return avgbatchacc, avgbatchloss
    
def plot_cm_final(model,data_iter):
    matrix = np.zeros((5, 5))
    for i, batch in enumerate(data_iter):
        batch_input, batch_input_length = batch.text
        outputs = model(batch_input, batch_input_length)
        _, preds = outputs.max(1)
        matrix += confusion_matrix(batch.label,preds,labels=range(5)) 
    
    classes = ['Academics', 'Alerts', 'Personal', 'Professional','Promotions and Events']
    cm = pd.DataFrame(matrix, index = [i for i in classes], columns = [c for c in classes])
    sn.heatmap(cm, annot=True)    
    
def decontracted(phrase): 
    """
    "Cleaning": Undoing contractions (e.g. can't --> can not)
    """
    # there could be cases I didn't think of. Feel free to add more
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def backtrans(directory): # !pip install googletrans in main 
    """
    Data augmentation (our English sample --> translate --> translate back) 
    """
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.startswith("a"):
            f = open(directory + filename, 'r')
            f = f.read()
            translator = Translator()
            try:
                # back translation to French
                imt = translator.translate(f, src='en',dest = 'fr')
                imt_text = imt.text
                result = translator.translate(imt.text,src='fr',dest = 'en')
                result_text = result.text

                with open(directory + "a_fr_" + filename, 'w') as out:
                    out.writelines(result_text)
            except:
                try: 
                    # back translation to German
                    imt = translator.translate(f, src='en',dest = 'de')
                    imt_text = imt.text
                    result = translator.translate(imt.text,src='de',dest = 'en')
                    result_text = result.text

                    with open(directory + "a_de_" + filename, 'w') as out:
                        out.writelines(result_text)
                except: 
                    with open(directory + filename, 'w') as out:
                        out.writelines(f)

    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.startswith("a"):
            f = open(directory + filename, 'r')
            f = f.read()
            translator = Translator()
            try:
                # back translation to Korean
                imt = translator.translate(f, src='en',dest = 'ko')
                imt_text = imt.text
                result = translator.translate(imt.text,src='ko',dest = 'en')
                result_text = result.text


                with open(directory + "a_ko_" + filename, 'w') as out:
                    out.writelines(result_text)
            except:
                try: 
                    # back translation to Japanese
                    imt = translator.translate(f, src='en',dest = 'ja')
                    imt_text = imt.text
                    result = translator.translate(imt.text,src='ja',dest = 'en')
                    result_text = result.text


                    with open(directory + "a_ja_" + filename, 'w') as out:
                        out.writelines(result_text)
                except: 
                    with open(directory + filename, 'w') as out:
                        out.writelines(f)
                         
def lexical_sub(directory, replace=0.1, sample=0.35): # !pip install nltk in main 
    """
    Data augmentation (Applies lexical substitution on a set of words in randomly picked txt files. Input directory should correspond to 
    the class-labelled folders in which the raw txt data is being stored. replace gives proportion of words to replace)
    """
    prefix = 'lex_'
    for file in os.listdir(directory):
        # Augment <sample> proportion of data
        if random.random() < sample:
            if file.endswith(".txt") and not file.startswith(prefix):
                text = []
                lines = open(os.path.join(directory, file), 'r', encoding='utf8').readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    words = line.split()
                    if words:
                        random_words = words.copy()
                        random_words = [word for word in random_words if len(word) > 2]
                        random.shuffle(random_words)

                        replace_num = len(random_words) * replace
                        num_replaced = 0
                        for random_word in random_words:
                            synonyms = set()
                            for syn in wordnet.synsets(random_word): 
                                for l in syn.lemmas(): 
                                    synonym = l.name().replace("_", " ").replace("-", " ").lower()
                                    synonyms.add(synonym)
                            if random_word in synonyms:
                                synonyms.remove(random_word)
                            if synonyms:
                                synonym = random.choice(list(synonyms))
                                words = [synonym if word == random_word else word for word in words]
                                num_replaced += 1
                            if num_replaced >= replace_num:
                                break

                    new_line = ' '.join(words)
                    text.append([new_line])
                with open(os.path.join(directory, prefix + file), 'w', encoding='utf8') as f:
                    writer = csv.writer(f)
                    writer.writerows(text) 
        break
        
                        
