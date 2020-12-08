import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
import nltk
import numpy as np

try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')

stemmer = PorterStemmer()
#stemmer = SnowballStemmer('english')
#%%
stopword_set = set(stopwords.words("english"))

data = pd.read_csv("encounter.csv").rename(str.lower, axis='columns')
soap = data['soap_note'].dropna().reset_index(drop=True)
soap_temp = [re.split('o:''|''o :', i) for i in soap] # split by "o:" or "o :"
temp_sentences = [i[0].strip().strip('s:').lower() for i in soap_temp]

document = '\n '.join(temp_sentences)

document = document.replace(r"'s", "  is")
document = document.replace(r"'ve", " have")
document = document.replace(r"can't", "cannot")
document = document.replace(r"musn't", "must not")
document = document.replace(r"n't", " not")
document = document.replace(r"i'm", "i am")
document = document.replace(r"'re", " are")
document = document.replace(r"'d", " would")
document = document.replace(r"\'ll", " will")
document = document.replace(r",", " ")
document = document.replace(r".", " . ")
document = document.replace(r"!", " ! ")
document = document.replace(r"/", " ")
document = document.replace(r"^", " ^ ")
document = document.replace(r"+", " + ")
document = document.replace(r"-", " - ")
document = document.replace(r"=", " = ")
document = document.replace(r"'", " ")
document = document.replace(r":", " : ")
document = document.replace(r" e g ", " eg ")
document = document.replace(r" b g ", " bg ")
document = document.replace(r" u s ", " united states ")
document = document.replace(r" 9 11 ", "911")
document = document.replace(r"e - mail", "email")
document = document.replace(r"e-mail", "email")
document = document.replace(r" e mail", "email")
document = document.replace(r"email", "email")
document = document.replace(r"j k", "jk")
document = document.replace(r"shoulda", "should have")
document = document.replace(r"coulda", "could have")
document = document.replace(r"woulda", "would have")
document = document.replace(r"http", "")
document = document.replace(r"c/o", "complaint of")
document = document.replace(r"h/o", "history of")
document = document.replace(r"yrs", "years")
document = document.replace(r"pmh", "past medical history")
document = document.replace(r"psh", "past surgical history")
document = document.replace(r"b/l", "bilateral")
document = document.replace(r"nkda", "no known drug allergies")
document = document.replace(r"crf", "chronic renal failure")
document = document.replace(r"arf", "acute renal failure")
document = document.replace(r"w/", "with")
document = document.replace(r" m ", " male ")
document = document.replace(r" f ", " female ")
document = document.replace(r" ys ", " years ")
document = document.replace(r" r ", " right ")
document = document.replace(r" rt ", " right ")
document = document.replace(r" l ", " left ")
document = document.replace(r" lt ", " left ")
document = document.replace(r" pt ", " patient ")
document = document.replace(r" yo ", " years old ")
document = document.replace(r" yr ", " years old ")

document = document.split('\n ')
if stemmer:
    for i, sentence in enumerate(document):
        document[i] = ' '.join([stemmer.stem(w) for w in sentence.split() if w not in stopword_set])
#%%
#sentences = [re.split(r'[-\s.,;!?]+', i) for i in temp_sentences]
#for i, token in enumerate(sentences):
 #  sentences[i] = [w for w in token if not w in stopword_set]

#%% n-grams
# One
#sentences = [re.split(r'[-\s.,;!?]+', i) for i in temp_sentences]

#corpus = {}
#for i, sentence in enumerate(sentences):
 #   corpus['sent' + str(i)] = dict((token, 1) for token in sentence)
#df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

# Two
#tokens_two = list(ngrams(i, 2) for i in sentences)
