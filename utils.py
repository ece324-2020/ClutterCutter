import os
import pandas as pd
import csv

def reformat_txt(path):
    """
    Combines text files for each data label under one tsv file.
    """
    for dir in os.listdir(path):
        if dir == "data.tsv":
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

# Accuracy function (for both models)
def accuracy(predictions, labels):
"""
Called in training loop (on training data), evaluate function (on evaluation data)
Called after training loop on evaluate function (on test data)
Returns decimal accuracy (0.0 - 1.0)
"""
    correct = 0
    ind = 0 
    for c in predictions:
        if (c.item()>0.5):
            r = 1.0
        else:
            r = 0.0
        
        if (r== labels[ind].item()):
            correct += 1
        ind += 1
    return (correct/len(labels))

# Evalutation function for baseline 
def evaluateBaseline(model, data_iter):
    """
    Called in training loop (on evaluation data)
    Called after training loop (on test data) 
    Returns decimal accuracy (0.0 - 1.0) and loss 
    
    """
    loss_fnc = nn.BCEWithLogitsLoss()
    correct = 0 
    
    batchloss_accum = 0.0
    batchacc_accum = 0.0
    
    for i,batch in enumerate(data_iter,0): # go through each batch in data_iter
        batch_input, batch_input_length = batch.text
        sentence_length = batch_input_length[0].item() # = len(batch_input)
        
        outputs = model(torch.reshape(batch_input,(sentence_length,len(batch)))) #batch has size [sentence length, batch size] <-- probably will need adjustment
        
        #Calculate accuracy 
        acc = accuracy(outputs,batch.label.float())
        batchacc_accum = batchacc_accum + acc
        
        #Calculate loss
        batchloss = loss_fnc(outputs, batch.label.float()) #(batch.label) (tensor of 64 1s and 0s)
        batchloss_accum = batchloss_accum + batchloss.item()

    avgbatchloss = batchloss_accum/len(data_iter)
    avgbatchacc = batchacc_accum/len(data_iter)
    return avgbatchacc, avgbatchloss

# Modified evaluation function for the RNN (changed to make model take in length too)
def evaluateRNN(model, data_iter):
    
    loss_fnc = nn.BCEWithLogitsLoss()
    correct = 0    
    
    batchloss_accum = 0.0
    batchacc_accum = 0.0
    
    for i,batch in enumerate(data_iter,0): # Go through each batch in data_iter
        batch_input, batch_input_length = batch.text
        sentence_length = batch_input_length[0].item() # = len(batch_input)
  
        outputs = model(batch_input,batch_input_length) #batch_input has size [sentence length, batch size]
        
        #Calculate accuracy 
        acc = accuracy(outputs,batch.label.float())
        batchacc_accum = batchacc_accum + acc
        
        #Calculate loss
        batchloss = loss_fnc(outputs, batch.label.float()) 
        batchloss_accum = batchloss_accum + batchloss.item()

    avgbatchloss = batchloss_accum/len(data_iter)
    avgbatchacc = batchacc_accum/len(data_iter)
    
    return avgbatchacc, avgbatchloss


# For testing purposes:
if __name__ == "__main__":
    data_path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data"
    reformat_txt(data_path)
    test_df = load_df(data_path)
    test_df.to_csv(os.path.join(data_path, 'data.tsv'), sep='\t', index=False)
