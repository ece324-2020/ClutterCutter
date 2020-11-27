# requires !pip install nltk
import os
import random
import csv

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
        break

if __name__ == "__main__":
    path = r"C:\Users\theow\Documents\Eng Sci Courses\Year 3\Fall Semester\ECE324\Project\data_test\Academics"
    nltk.download('wordnet')
    lexical_sub(path)