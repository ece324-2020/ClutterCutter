import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

from utils import *
from models import *

def main():
    # Data processing
    reformat_txt(data_path)
    pre_processing(data_path)
    train_iter, val_iter, test_iter, vocab = make_iter(data_path, batch_size)

    # Initiate model 
    if network == 'baseline':
        model = Baseline(embedding_dim, vocab) 
    elif network =='rnn':
        model = RNN(embedding_dim, vocab, hidden_dim)
    else:
        raise ValueError('Invalid network chosen')
    
    # Define loss and optimzer functions 
    loss_fnc = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Store for plotting
    loss_list = []
    acc_list = []
    nepoch = []

    val_acc_list = []
    val_loss_list = []

    # Training loop
    for epoch in range(num_epochs): 
        nepoch = nepoch + [epoch]

        running_loss = 0.0
        running_acc = 0.0
        model.train() 
        
        for i, batch in enumerate(train_iter):

            optimizer.zero_grad()
            batch_input, batch_input_length = batch.text

            outputs = model(batch_input, batch_input_length)
            
            # Compute loss
            loss = loss_fnc(outputs, batch.label) 
            running_loss += loss.item() 
            
            # Update gradients
            loss.backward()
            optimizer.step()
            
            # Compute accuracy 
            acc = accuracy(outputs,batch.label)
            running_acc += acc
            
        model.eval()
        train_acc = running_acc/len(train_iter)
        train_loss = running_loss/len(train_iter)
        vacc, vloss = evaluate(model, val_iter)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("Average Loss: ", train_loss)
        print("Average Accuracy:", train_acc)
        print("Validation Loss: ", vloss)
        print("Validation Accuracy: ", vacc)

        loss_list.append(train_loss)
        acc_list.append(train_acc)
        val_acc_list.append(vacc)   
        val_loss_list.append(vloss)
    
    model.eval()
    tacc,tloss = evaluate(model, test_iter)

    print(f"Final Test Acccuracy: {tacc}")
    print(f"Final Test Loss: {tloss}")
    
    #Plot Losses
    plt.plot(nepoch,loss_list, label = 'Train')
    plt.plot(nepoch,val_loss_list, label = 'Valid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title("Training and Validation Loss")
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show() 

    #Plot Accuracies
    plt.plot(nepoch,acc_list, label = 'Train')
    plt.plot(nepoch,val_acc_list, label = 'Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy") 
    plt.title("Training and Validation Accuracy")
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show() 

if __name__ == '__main__':
    torch.manual_seed(0)
    # Move to config object/file!
    data_path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data"
    batch_size = 20
    learning_rate = 0.01
    num_epochs = 10
    embedding_dim = 100
    hidden_dim = 100
    network = 'rnn'

    main()



