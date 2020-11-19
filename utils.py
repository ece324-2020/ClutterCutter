import os
import pandas as pd
import csv

def reformat_txt(path):
    """
    Combines text files for each data label under one tsv file.
    """
    for dir in os.listdir(path):
        text = []
        tsv_path = os.path.join(path, dir, "text.tsv")

        # remove previous copy of tsv
        if os.path.exists(tsv_path):
            os.remove(tsv_path)

        # Extract text from each file
        for file in os.listdir(os.path.join(path, dir)):
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
        label = labels[dir]
        data_path = os.path.join(path, dir, "text.tsv")
        text_df = pd.read_csv(data_path, sep='\t')
        text_df.insert(1, 'label', label) # Adds label based on directory name
        df = pd.concat([df, text_df])
    df = df.reset_index(drop=True)
    return df

# For testing purposes:
if __name__ == "__main__":
    data_path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data_test"
    reformat_txt(data_path)
    test_df = load_df(data_path)
    test_df.to_csv(os.path.join(data_path, 'data.tsv'), sep='\t', index=False)
