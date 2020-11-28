# Requires nltk and google translate packages
# !pip install googletrans
# !pip install nltk

import os

import random
import string
import re 
import csv

from zipfile import ZipFile
from itertools import chain
from glob import glob
from googletrans import Translator
import pandas as pd

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import wordnet

def lexical_sub(directory, replace=0.1, sample=0.35):
    """
    Applies lexical substitution on a set of words in randomly picked txt files. Input directory should correspond to 
    the class-labelled folders in which the raw txt data is being stored. replace gives proportion of words to replace
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

def backtrans(directory, sample=0.35):
    for filename in os.listdir(directory):
        if random.random() < sample:
            if filename.endswith(".txt") and not filename.startswith("a"):
                f = open(os.path.join(directory, filename), 'r', encoding='utf8')
                f = f.read()
                translator = Translator()
                try:
                    # back translation to French
                    imt = translator.translate(f, src='en',dest = 'fr')
                    imt_text = imt.text
                    result = translator.translate(imt.text,src='fr',dest = 'en')
                    result_text = result.text

                    with open(os.path.join(directory, "a_fr_" + filename), 'w', encoding='utf8') as out:
                        out.writelines(result_text)
                except:
                    try: 
                        # back translation to German
                        imt = translator.translate(f, src='en',dest = 'de')
                        imt_text = imt.text
                        result = translator.translate(imt.text,src='de',dest = 'en')
                        result_text = result.text

                        with open(os.path.join(directory, "a_de_" + filename), 'w', encoding='utf8') as out:
                            out.writelines(result_text)
                    except: 
                        with open(os.path.join(directory, filename), 'w', encoding='utf8') as out:
                            out.writelines(f)

    for filename in os.listdir(directory):
        if random.random() < sample:
            if filename.endswith(".txt") and not filename.startswith("a"):
                f = open(os.path.join(directory, filename), 'r', encoding='utf8')
                f = f.read()
                translator = Translator()
                try:
                    # back translation to Korean
                    imt = translator.translate(f, src='en',dest = 'ko')
                    imt_text = imt.text
                    result = translator.translate(imt.text,src='ko',dest = 'en')
                    result_text = result.text


                    with open(os.path.join(directory, "a_ko_" + filename), 'w', encoding='utf8') as out:
                        out.writelines(result_text)
                except:
                    try: 
                        # back translation to Japanese
                        imt = translator.translate(f, src='en',dest = 'ja')
                        imt_text = imt.text
                        result = translator.translate(imt.text,src='ja',dest = 'en')
                        result_text = result.text


                        with open(os.path.join(directory, "a_ja_" + filename), 'w', encoding='utf8') as out:
                            out.writelines(result_text)
                    except: 
                        with open(os.path.join(directory, filename), 'w', encoding='utf8') as out:
                            out.writelines(f)

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

def pre_processing(path, sample=0.35):
    # Create new augmented samples:
    for dir in os.listdir(path):
        if dir != ".DS_Store" and dir[-4:] != '.tsv':
            directory = os.path.join(path, dir)
            lexical_sub(directory, sample=sample)
            backtrans(directory, sample=sample)

    reformat_txt(path) # Creates text.tsv
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

if __name__ == "__main__":
    data_path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data_test"
    pre_processing(data_path)