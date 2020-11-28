import torch
import torch.nn as nn
import numpy as np

import torchtext
from torchtext import data
import spacy

import os
import pandas as pd

from sklearn.metrics import confusion_matrix

import seaborn as sn

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