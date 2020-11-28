import re
from collections import Counter

filename = 'personal.tsv'
with open(filename) as f:
    passage = f.read()

words = re.findall(r'\w+', passage)

cap_words = [word.lower() for word in words if len(word)>4 and word not in ['there','these']]

word_counts = Counter(cap_words)

print(word_counts)
