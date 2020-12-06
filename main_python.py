import os
import pandas as pd
import numpy as np

import re

#import matplotlib.pyplot as plt
#import seaborn as sns

data = pd.read_csv("encounter.csv").rename(str.lower,axis='columns')
#e_dx = pd.read_csv("encounter_dx.csv").rename(str.lower,axis='columns')
#lr = pd.read_csv("lab_results.csv").rename(str.lower,axis='columns')
#mf = pd.read_csv("medication_fulfillment.csv").rename(str.lower,axis='columns')

soap = data['soap_note'].dropna()
temp_sentences = [i.split('o:')[0].strip().strip('s:').lower() for i in soap]
sentences = [re.split(r'[-\s.,;!?]+', i) for i in temp_sentences]

ts = sentences[0]
vocab = sorted(set(ts))
one_hot = np.zeros((len(ts), len(vocab)), int)
for i, word in enumerate(ts):
    one_hot[i,vocab.index(word)] = 1
print(one_hot)
pd.DataFrame(one_hot, columns=vocab).head()

df = pd.DataFrame(pd.Series(dict([(token,1) for token in ts])), columns=['sent']).T
df

corpus = {}
for i, sent in enumerate(sentences):
    corpus['sent' + str(i)] = dict((token,1) for token in sent.split())

df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T