import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import string
import re 

def decontracted(df):
    # specific
    df['text'] = df['text'].str.replace('won\'t', 'will not')
    df['text'] = df['text'].str.replace('can\'t', 'can not')

    # general
    df['text'] = df['text'].str.replace('\'t', ' not')
    df['text'] = df['text'].str.replace('\'re', ' are')
    df['text'] = df['text'].str.replace('\'s', ' is')
    df['text'] = df['text'].str.replace('\'d', ' would')
    df['text'] = df['text'].str.replace('\'ll', ' will')
    df['text'] = df['text'].str.replace('\'ve', ' have')
    df['text'] = df['text'].str.replace('\'m', ' am')
    return df

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
                        content = f.read()
                        content.strip('\t') # removes tabs so content can be stored in tsv format.
                        text.append([content])

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
    data = decontracted(data)
    data = strip(data) # Remove 

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

# Evalutation function for baseline 
def evaluateBaseline(model, data_iter):
    """
    Called in training loop (on evaluation data)
    Called after training loop (on test data) 
    Returns decimal accuracy (0.0 - 1.0) and loss 
    
    """
    loss_fnc = nn.CrossEntropyLoss()
    correct = 0 
    
    batchloss_accum = 0.0
    batchacc_accum = 0.0
    
    for i,batch in enumerate(data_iter,0): # go through each batch in data_iter
        batch_input, batch_input_length = batch.text
        sentence_length = batch_input_length[0].item() # = len(batch_input)
        
        outputs = model(torch.reshape(batch_input,(sentence_length,len(batch)))) #batch has size [sentence length, batch size] <-- probably will need adjustment
        
        #Calculate accuracy 
        acc = accuracy(outputs,batch.label)
        batchacc_accum = batchacc_accum + acc
        
        #Calculate loss
        batchloss = loss_fnc(outputs, batch.label) #(batch.label) (tensor of 64 1s and 0s)
        batchloss_accum = batchloss_accum + batchloss.item()

    avgbatchloss = batchloss_accum/len(data_iter)
    avgbatchacc = batchacc_accum/len(data_iter)
    
    return avgbatchacc, avgbatchloss

# Modified evaluation function for the RNN (changed to make model take in length too)
def evaluateRNN(model, data_iter):
    
    loss_fnc = nn.CrossEntropyLoss()
    correct = 0    
    
    batchloss_accum = 0.0
    batchacc_accum = 0.0
    
    for i,batch in enumerate(data_iter,0): # Go through each batch in data_iter
        batch_input, batch_input_length = batch.text
        sentence_length = batch_input_length[0].item() # = len(batch_input)
  
        outputs = model(batch_input,batch_input_length) #batch_input has size [sentence length, batch size]
        
        #Calculate accuracy 
        acc = accuracy(outputs,batch.label)
        batchacc_accum = batchacc_accum + acc
        
        #Calculate loss
        batchloss = loss_fnc(outputs, batch.label) 
        batchloss_accum = batchloss_accum + batchloss.item()

    avgbatchloss = batchloss_accum/len(data_iter)
    avgbatchacc = batchacc_accum/len(data_iter)
    
    return avgbatchacc, avgbatchloss


# For testing purposes:
if __name__ == "__main__":
    data_path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data"
    reformat_txt(data_path)
    pre_processing(data_path)
