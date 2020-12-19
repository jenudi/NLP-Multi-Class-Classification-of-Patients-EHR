import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from collections import OrderedDict
from collections import Counter
import copy
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
stopword_set = set(stopwords.words("english"))


encounter_data = pd.read_csv("encounter.csv").rename(str.lower, axis='columns')
encounter_dx = pd.read_csv("encounter_dx.csv").rename(str.lower, axis='columns')
lab_results = pd.read_csv("lab_results.csv").rename(str.lower, axis='columns')

data1=encounter_data[['encounter_id','member_id', 'patient_gender', 'has_appt', 'soap_note']].set_index('encounter_id').sort_index()
data2=encounter_dx.groupby('encounter_id')['code', 'description', 'severity'].apply(lambda x: list(x.values)).sort_index()
data3=lab_results.groupby('encounter_id')['result_name','result_description','numeric_result', 'units'].apply(lambda x: list(x.values)).sort_index()

data=pd.concat([data1,data2,data3],axis=1)
data=data.rename(columns={0:'code_description_severity', 1:'result_name_result-description_numeric-result_units'})

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
document = document.replace(r"pt", " pt ")
document = document.replace(r"(", "")
document = document.replace(r")", "")
document = document.replace(r"=", "")
#document = document.replace(r"/", " ")
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
document = document.replace(r"c/o", "complains of")
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
document = document.replace(r" x ", " times ")

document = document.split('\n ')

if stemmer:
    for i, sentence in enumerate(document):
        document[i] = ' '.join([stemmer.stem(w) for w in sentence.split() if w not in stopword_set])

#%%

sentences = [re.split(r'[-\s.,;!?]+', i)[:-1] for i in document]

#corpus = {}
#for i, sentence in enumerate(sentences):
#    corpus['sent' + str(i)] = Counter(sentence)
#df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T


#%%
doc_tokens = []
for sent in sentences:
    doc_tokens += [sorted(sent)]
lexicon = sorted(set(sum(doc_tokens, [])))

zero_vector = OrderedDict((token, 0) for token in lexicon)
document_tfidf_vectors = []

for sent in sentences:
    vec = copy.copy(zero_vector)
    tokens = sent
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        sents_containing_key = 0
        for _sent in sentences:
            if key in  _sent:
                sents_containing_key += 1
        tf = value / len(lexicon)
        if sents_containing_key:
            #idf = len(sentences) / sents_containing_key
            idf = (1+np.log((1+len(sentences)) / (1+sents_containing_key)))
        else:
            idf = 0
        vec[key] = tf * idf
    document_tfidf_vectors.append(vec)

corpus = {}
for i, n in enumerate(document_tfidf_vectors):
    corpus['sent' + str(i)] = n
df = pd.DataFrame.from_records(corpus).T

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = {}
for i, sentence in enumerate(sentences):
    corpus['sent' + str(i)] = ' '.join(sentence)

tfidf = TfidfVectorizer(min_df=0.0,smooth_idf=True)
tfs = tfidf.fit_transform(corpus.values())

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

