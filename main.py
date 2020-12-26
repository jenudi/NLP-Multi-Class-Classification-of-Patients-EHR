#%% Imports

import pandas as pd
import numpy as np
import re
import random
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


from nltk.stem import SnowballStemmer
from collections import Counter
import copy
from nltk.util import ngrams
import nltk

#%% Loading Data

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


#%% Pre-processing

class document:

    def __init__(self,text):
        self.text=text
        self.sentences=[]
        self.train = doc_set([])
        self.validation = doc_set([])
        self.test = doc_set([])

    def do_replaces(self):
        self.text = self.text.replace(r"'s", "  is")
        self.text = self.text.replace(r"'ve", " have")
        self.text = self.text.replace(r"can't", "cannot")
        self.text = self.text.replace(r"musn't", "must not")
        self.text = self.text.replace(r"n't", " not")
        self.text = self.text.replace(r"i'm", "i am")
        self.text = self.text.replace(r"'re", " are")
        self.text = self.text.replace(r"'d", " would")
        self.text = self.text.replace(r"\'ll", " will")
        self.text = self.text.replace(r",", " ")
        self.text = self.text.replace(r".", " . ")
        self.text = self.text.replace(r"!", " ! ")
        self.text = self.text.replace(r"pt", " pt ")
        self.text = self.text.replace(r"(", "")
        self.text = self.text.replace(r")", "")
        self.text = self.text.replace(r"=", "")
        self.text = self.text.replace(r"^", " ^ ")
        self.text = self.text.replace(r"+", " + ")
        self.text = self.text.replace(r"-", " - ")
        self.text = self.text.replace(r"=", " = ")
        self.text = self.text.replace(r"'", " ")
        self.text = self.text.replace(r":", " : ")
        self.text = self.text.replace(r" e g ", " eg ")
        self.text = self.text.replace(r" b g ", " bg ")
        self.text = self.text.replace(r" u s ", " united states ")
        self.text = self.text.replace(r" 9 11 ", "911")
        self.text = self.text.replace(r"e - mail", "email")
        self.text = self.text.replace(r"e-mail", "email")
        self.text = self.text.replace(r" e mail", "email")
        self.text = self.text.replace(r"email", "email")
        self.text = self.text.replace(r"j k", "jk")
        self.text = self.text.replace(r"shoulda", "should have")
        self.text = self.text.replace(r"coulda", "could have")
        self.text = self.text.replace(r"woulda", "would have")
        self.text = self.text.replace(r"http", "")
        self.text = self.text.replace(r"c/o", "complains of")
        self.text = self.text.replace(r"h/o", "history of")
        self.text = self.text.replace(r"yrs", "years")
        self.text = self.text.replace(r"pmh", "past medical history")
        self.text = self.text.replace(r"psh", "past surgical history")
        self.text = self.text.replace(r"b/l", "bilateral")
        self.text = self.text.replace(r"nkda", "no known drug allergies")
        self.text = self.text.replace(r"crf", "chronic renal failure")
        self.text = self.text.replace(r"arf", "acute renal failure")
        self.text = self.text.replace(r"w/", "with")
        self.text = self.text.replace(r" m ", " male ")
        self.text = self.text.replace(r" f ", " female ")
        self.text = self.text.replace(r" ys ", " years ")
        self.text = self.text.replace(r" r ", " right ")
        self.text = self.text.replace(r" rt ", " right ")
        self.text = self.text.replace(r" l ", " left ")
        self.text = self.text.replace(r" lt ", " left ")
        self.text = self.text.replace(r" pt ", " patient ")
        self.text = self.text.replace(r" yo ", " years old ")
        self.text = self.text.replace(r" yr ", " years old ")
        self.text = self.text.replace(r" x ", " times ")

    def make_sentences(self,char):
        self.sentences = [sentence(sent) for sent in self.text.split(char)]

    def get_sentences(self):
        return [sent.text for sent in self.sentences]

    def train_test_split(self, validation=20,test=30):
        random.shuffle(self.sentences)
        train_len = len(self.sentences)-(validation+test)
        self.train = doc_set(self.sentences[0:train_len])
        self.validation = doc_set(self.sentences[train_len:train_len+validation])
        self.test = doc_set(self.sentences[train_len+validation:])


class doc_set:

    def __init__(self, sentences):
        self.sentences = sentences
        self.lexicon = []
        tfidf=[]

    def get_sentences(self):
        return [sent.text for sent in self.sentences]

    def make_lexicon(self):
        doc_tokens=[]
        for sent in self.sentences:
            doc_tokens += [sorted(sent.get_one_gram_tokens())]
        self.lexicon = sorted(set(sum(doc_tokens, [])))

    def get_zero_vector(self):
        return OrderedDict((tok, 0) for tok in self.lexicon)

    def make_tfidf(self):
        tfidf = TfidfVectorizer(min_df=0.0, smooth_idf=True, norm='l1')
        self.tfidf = tfidf.fit_transform(self.get_sentences())

    def clusters_to_sentences_indexes_dict(self,clusters,num_of_clusters):
        dict={}
        for cluster_num in range(num_of_clusters):
            true_clusters=clusters==cluster_num
            dict[cluster_num]=[index for index, truth_value in enumerate(true_clusters) if truth_value]
        return dict





class sentence:

    def __init__(self,text):
        self.text=text
        self.one_gram_tokens=[]

    def stem_and_check_stop(self,stopword_set):
        self.text = ' '.join([stemmer.stem(w) for w in self.text.split() if w not in stopword_set])

    def make_one_gram_tokens(self):
        self.one_gram_tokens = [token(tok) for tok in re.split(r'[-\s.,;!?]+', self.text)[:-1]]

    def get_one_gram_tokens(self):
        return [tok.text for tok in self.one_gram_tokens]


class token:
    def __init__(self,text):
        self.text=text


try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')

stopword_set = set(stopwords.words("english"))
stemmer = PorterStemmer()
#stemmer = SnowballStemmer('english')

doc = document('\n '.join(temp_sentences))
doc.do_replaces()
doc.make_sentences('\n ')

for sent in doc.sentences:
    sent.stem_and_check_stop(stopword_set)

for sent in doc.sentences:
    sent.make_one_gram_tokens()

for sent in doc.sentences:
    sent.text=' '.join(sent.get_one_gram_tokens())

#corpus = {}
#for i, sentence in enumerate(tokens):
#    corpus['sent' + str(i)] = Counter(sentence)
#df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

#%% TF-IDF
##lexicon is in ordercompare which words exist in other lexicons

doc.train_test_split()
doc.train.make_lexicon()
doc.train.make_tfidf()

#%% DBSCAN
'''
epsilons=[0.1:0.1:1]
clustering = DBSCAN(eps=3, min_samples=50).fit(doc.train.tfidf.todense())
print("number of groups: " + str())
'''
#%% K-means
#n_clusters=range(5,)
num_of_clusters=10
kmeans = KMeans(n_clusters=num_of_clusters, random_state=0).fit(doc.train.tfidf.todense())
dict=doc.train.clusters_to_sentences_indexes_dict(kmeans.labels_,num_of_clusters)
#for cluster_num in range(num_of_clusters):
for cluster_num in [1]:
    print("current cluster "+ str(cluster_num))
    for sent_i in dict[cluster_num]:
        print(doc.train.get_sentences()[sent_i])


#%% SVD
## maybe change to t-SNE
U, s, Vt = np.linalg.svd(tfs_dataframe.T)
S = np.zeros((len(U), len(Vt)))
pd.np.fill_diagonal(S, s)
pd.DataFrame(S).round(1)

#%% SVD err
err = []
for numdim in range(len(s), 0, -1):
    S[numdim - 1, numdim - 1] = 0
    tfs_dataframe_reconstructed = U.dot(S).dot(Vt)
    err.append(np.sqrt(((tfs_dataframe_reconstructed - tfs_dataframe.T).values.flatten() ** 2).sum() / np.product(tfs_dataframe.T.shape)))
np.array(err).round(2)

ts = ts.cumsum()
ts.plot()
plt.show()

#%% Word2Vec
#if some words in our lexicon dont exist in words2vec lexicon change them to 'unknown' token
#check how to determine number of features
num_features = 300
min_word_count = 0
num_workers = 2
window_size = 5
subsampling = 1e-3

model = Word2Vec(doc.get_sentences(), min_count=0,size=num_features,workers=2, window =5, sg = 0, sample= 1e-3)


#%% TF-IDF Manual
"""
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
"""
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

# Two-grams
#tokens_two = list(ngrams(i, 2) for i in sentences)