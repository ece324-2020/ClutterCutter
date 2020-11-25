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
                          n = re.sub(r'[^\w\s]','',n)
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
                        phrase = phrase.lower()
                        phrase.strip('\t') # removes tabs so content can be stored in tsv format.
                        phrase = ' '.join([w for w in phrase.split() if len(w) > 2])
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

def strip(df):
    df['text'] = df['text'].str.replace(r'[^\w\s]+', ' ') # replaces all punctuation with spaces
    df['text'] = df['text'].str.replace('\n', ' ') # remove new lines
    return df

def pre_processing(path):
    data = load_df(path) # Load labelled dataframe
    data = strip(data) 

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
    
def plot_cm(model, data_iter):
    cm = np.zeros((5, 5))
    for i, batch in enumerate(data_iter):
        batch_input, batch_input_length = batch.text
        outputs = model(batch_input, batch_input_length)
        _, preds = outputs.max(1)
        cm += confusion_matrix(batch.label, preds, labels=range(5))

    print("Confusion Matrix")
    print(cm)

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

