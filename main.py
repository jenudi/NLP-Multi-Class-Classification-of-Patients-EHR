import pandas as pd
import numpy as np
import re
from nltk.util import ngrams

data = pd.read_csv("encounter.csv").rename(str.lower, axis='columns')
soap = data['soap_note'].dropna()
temp_sentences = [i.split('o:')[0].strip().strip('s:').lower() for i in soap]

# One n-grams

sentences = [re.split(r'[-\s.,;!?]+', i) for i in temp_sentences]

corpus = {}
for i, sent in enumerate(sentences):
    corpus['sent' + str(i)] = dict((token, 1) for token in sent)
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

# Two n-grams

sentences = 