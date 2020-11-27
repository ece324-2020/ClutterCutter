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

from torchtext.vocab import FastText

def main():
    torch.manual_seed(0)

    # Incorporate Weights and Biases tracking
    if wandb_toggle:
        run = wandb.init(project="ClutterCutter", reinit=True)
        wandb.config.update({"epoch": num_epochs, 
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "architecture": network,
                            "hidden_dim": hidden_dim}, allow_val_change=True)

    # Data processing
    if os.path.exists(os.path.join(data_path, 'train.tsv')):
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
            acc = accuracy(outputs, batch.label)
            running_acc += acc
            
        model.eval()
        train_acc = running_acc/len(train_iter)
        train_loss = running_loss/len(train_iter)
        val_acc, val_loss = evaluate(model, val_iter)
        if wandb_toggle:
            wandb.log({"train_acc": train_acc, 
                       "train_loss": train_loss,
                       "val_acc": val_acc,
                       "val_loss": val_acc})
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("Average Loss: ", train_loss)
        print("Average Accuracy:", train_acc)
        print("Validation Loss: ", val_loss)
        print("Validation Accuracy: ", val_acc)
        print("\n")

        loss_list.append(train_loss)
        acc_list.append(train_acc)
        val_acc_list.append(val_acc)   
        val_loss_list.append(val_loss)
    
    model.eval()
    tacc,tloss = evaluate(model, test_iter)
    if wandb_toggle:
        wandb.log({"test_acc": tacc,
                   "test_loss": tloss})

    print(f"Final Test Acccuracy: {tacc}")
    print(f"Final Test Loss: {tloss}")
    
    #Plot Losses
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(nepoch,loss_list, label = 'Train')
    plt.plot(nepoch,val_loss_list, label = 'Valid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title("Training and Validation Loss")
    plt.legend(['Training', 'Validation'], loc='upper left')
    if wandb_toggle:
        wandb.log({"Loss Curves": wandb.Image(fig)})

    #Plot Accuracies
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(nepoch,acc_list, label = 'Train')
    plt.plot(nepoch,val_acc_list, label = 'Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy") 
    plt.title("Training and Validation Accuracy")
    plt.legend(['Training', 'Validation'], loc='upper left')
    if wandb_toggle:
        wandb.log({"Accuracy Curves": wandb.Image(fig)})
    
    plot_cm_final(model, test_iter) # Nicer matrix 

    run.finish()
    # Maybe code to save? Early stopping?

if __name__ == '__main__':
    data_path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data" 
    num_epochs = 30
    embedding_dim = 300  # (100 for GloVe, 300 for FastText)
    wandb_toggle = True # To enable or disable wandb tracking.
    network = 'rnn'

    # Hyperparameter grid search
    for learning_rate in [0.01, 0.005, 0.001]:
        for batch_size in [8, 16, 32, 64]:
            for hidden_dim in [50, 100, 200]:
                main()



