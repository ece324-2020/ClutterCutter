import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os


#from models import * #in separate models.py file

import torch.nn as nn
from torch.autograd import Variable
import time

import matplotlib.pyplot as plt

#Hyperparameters 
learning_rate = 0.01
batch_size = 20
epochs = 25
seed = 0

torch.manual_seed(seed)

# ------------------------------ BASELINE (FULL DATASET) ------------------------------ 
def main():
    # Instantiates 2 data.Field objects 
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)
    
    # Load the train, validation, and test datasets to become datasets
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='datawang/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # Create an object that can be enumerated (for training loop later)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
    sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    
    # Vocab object contains the index/token for each unique word in the dataset (looks through all sentences in dataset)
    TEXT.build_vocab(train_data, val_data, test_data)

    # Loading GloVe Vector and Using Embedding Layer
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape) #number of unique words 
    
    # Training the baseline model --------------------
    # Reproducability 
    torch.manual_seed(seed)

    # Initiate model 
    model = Baseline(100,vocab) ### 
    
    # Define loss and optimzer functions 
    loss_fnc = nn.CrossEntropyLoss()# Convert labels to one-hot to calculate loss 
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

    # Store for plotting
    loss_list = []
    acc_list = []
    nepoch = []

    val_acc_list = []
    val_loss_list = []

    # TRAINING LOOP --------------------
    for e in range(epochs): 
        nepoch = nepoch + [e]

        batchloss_accum = 0.0
        batchacc_accum = 0.0
        model.train() 

        # For batch in train_iter: #len train_iter is number of batches 
        for i, batch in enumerate(train_iter, 0):
            correct = 0 
            total = 0 
            
            # Zero parameter gradients
            optimizer.zero_grad()
            
            # Run model on inputs
            batch_input, batch_input_length = batch.text

            outputs = model(batch_input)
            #print(batch.label.float().shape)
            
            # Compute loss
            batchloss = loss_fnc(outputs, batch.label) 
            batchloss_accum = batchloss_accum + batchloss.item() #added values of loss for all batches
            #print('batchloss',batchloss)
            
            batchloss.backward()
            optimizer.step()
            
            # Compute accuracy 
            batchacc = accuracy(outputs,batch.label)
            batchacc_accum = batchacc_accum + batchacc
            #print("Batch accuracy",batchacc)
            
            if i == len(train_iter)-1: #len(trainloader) is len(dataset)
                model.eval()
                vacc, vloss = evaluateBaseline(model,val_iter)
                
                
                print("avg acc/epoch", batchacc_accum/len(train_iter))
                print('[%d, %5d] avg loss/epoch: %.3f' % (e + 1, i + 1, batchloss_accum/len(train_iter)))
                print("validation loss:", vloss)
                print("validation acc:", vacc)

                loss_list = loss_list + [batchloss_accum/len(train_iter)]
                acc_list = acc_list + [batchacc_accum/len(train_iter)]
                val_acc_list.append(vacc)   
                val_loss_list.append(vloss)

                batchloss_accum = 0.0
                batchacc_accum = 0.0 
    
    # Evaluate with test dataset
    model.eval()
    tacc,tloss = evaluateBaseline(model,test_iter)
    print(tacc,tloss)

    print("Final Test Acccuracy:", tacc)
    
    #LOSS TOGETHER
    plt.plot(nepoch,loss_list, label = 'Train')
    plt.plot(nepoch,val_loss_list, label = 'Valid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title("Training vs. Validation Loss Curve for full dataset")
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show() 

    #ACCURACIES TOGETHER
    plt.plot(nepoch,acc_list, label = 'Train')
    plt.plot(nepoch,val_acc_list, label = 'Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy") 
    plt.title("Training vs. Validation Accuracy Curve for full dataset")
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show() 
    
    #torch.save(model,'models/model_baseline.pt')


if __name__ == '__main__':
    main()



