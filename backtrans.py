# Remember to install the google trans library
# !pip install googletrans
from googletrans import Translator

import os
from zipfile import ZipFile
from itertools import chain
from glob import glob
import re

# I was testing this on the personal dataset
# change path and file names as needed 

directory = "Personal/"
def backtrans(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.startswith("a"):
            f = open(directory + filename, 'r')
            f = f.read()
            translator = Translator()
            try:
                # back translation to French
                imt = translator.translate(f, src='en',dest = 'fr')
                imt_text = imt.text
                result = translator.translate(imt.text,src='fr',dest = 'en')
                result_text = result.text

                with open(directory + "a_fr_" + filename, 'w') as out:
                    out.writelines(result_text)
            except:
                try: 
                    # back translation to German
                    imt = translator.translate(f, src='en',dest = 'de')
                    imt_text = imt.text
                    result = translator.translate(imt.text,src='de',dest = 'en')
                    result_text = result.text

                    with open(directory + "a_de_" + filename, 'w') as out:
                        out.writelines(result_text)
                except: 
                    with open(directory + filename, 'w') as out:
                        out.writelines(f)

    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.startswith("a"):
            f = open(directory + filename, 'r')
            f = f.read()
            translator = Translator()
            try:
                # back translation to Korean
                imt = translator.translate(f, src='en',dest = 'ko')
                imt_text = imt.text
                result = translator.translate(imt.text,src='ko',dest = 'en')
                result_text = result.text


                with open(directory + "a_ko_" + filename, 'w') as out:
                    out.writelines(result_text)
            except:
                try: 
                    # back translation to Japanese
                    imt = translator.translate(f, src='en',dest = 'ja')
                    imt_text = imt.text
                    result = translator.translate(imt.text,src='ja',dest = 'en')
                    result_text = result.text


                    with open(directory + "a_ja_" + filename, 'w') as out:
                        out.writelines(result_text)
                except: 
                    with open(directory + filename, 'w') as out:
                        out.writelines(f)
# function call
# backtrans(directory)
