import os
from zipfile import ZipFile
from itertools import chain
from glob import glob
import re

# I was testing this on the personal dataset
# change path and file names as needed 

Email_Hu = 'Personal.zip'
with ZipFile(Email_Hu, 'r') as zip:
  zip.extractall()
  print('Done loading my data')

# start here if you are not using zipped files 
directory = "Personal/"

def decontracted(phrase):
    # there could be cases I didn't think of. Feel free to add more
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        f = open(directory + filename, 'r')
        # lower case
        lines = [line.lower() for line in f]
        # remove numbers
        lines = ''.join(c for c in lines if not c.isdigit())
        # decontract 
        lines = ' '.join(decontracted(c) for c in lines.split())
        # remove words of length less than 2
        lines = ' '.join([w for w in lines.split() if len(w) > 2])

        with open(filename, 'w') as out:
            out.writelines(lines)
