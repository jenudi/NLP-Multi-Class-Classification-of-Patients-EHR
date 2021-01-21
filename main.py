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
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import SnowballStemmer
from collections import Counter
import copy
from nltk.util import ngrams
import nltk


#%% Classes and functions

class document:

    def __init__(self,text):
        self.text=text
        self.sentences=list()
        self.train = document_set(list())
        self.validation = document_set(list())
        self.test = document_set(list())

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
        self.text = self.text.replace(r" sym ", " symptom ")

    def make_sentences(self,char):
        self.sentences = [sentence_in_document(sentence) for sentence in self.text.split(char)]

    def get_sentences(self):
        return [sentence.text for sentence in self.sentences]

    def train_test_split(self, validation=20,test=30):
        random.shuffle(self.sentences)
        train_len = len(self.sentences)-(validation+test)
        self.train = document_set(self.sentences[0:train_len])
        self.validation = document_set(self.sentences[train_len:train_len+validation])
        self.test = document_set(self.sentences[train_len+validation:])


class document_set:

    def __init__(self, sentences):
        self.sentences = sentences
        self.lexicon = list()
        self.tfidf = list()
        self.word2vec = dict()

    def get_sentences(self):
        return [sentence.text for sentence in self.sentences]

    def get_original_sentences(self):
        return [sentence.original_text for sentence in self.sentences]

    def get_sentences_tokens(self):
        return [sentence.one_gram_tokens for sentence in self.sentences]

    def make_lexicon(self):
        doc_tokens=list()
        for sentence in self.sentences:
            doc_tokens += [sorted(sentence.one_gram_tokens)]
        self.lexicon = sorted(set(sum(doc_tokens, [])))

    def get_zero_vector(self):
        return OrderedDict((tok, 0) for tok in self.lexicon)

    def make_tfidf(self,model):
        self.tfidf = model.transform(self.get_sentences()).todense()

    def make_word2vec(self,word2vec_model,hyperparameter_lambda):
        self.word2vec[hyperparameter_lambda]=list()
        for sentence_tokens in self.get_sentences_tokens():
            word_embeddings= np.array([word2vec_model.wv[token] for token in sentence_tokens])
            word_embeddings_with_lambda=np.mean(word_embeddings,axis=0)*(len(word_embeddings)**hyperparameter_lambda)
            word_embeddings_with_lambda_normalised=word_embeddings_with_lambda/np.linalg.norm(word_embeddings_with_lambda)
            self.word2vec[hyperparameter_lambda].append(word_embeddings_with_lambda_normalised)

    def clusters_to_sentences_indexes_dict(self,clusters,num_of_clusters):
        dict={}
        for cluster_num in range(num_of_clusters):
            true_clusters = (clusters==cluster_num)
            dict[cluster_num]=[index for index, truth_value in enumerate(true_clusters) if truth_value]
        return dict


class sentence_in_document:

    def __init__(self,text):
        self.text=text
        self.original_text = text
        self.one_gram_tokens=[]

    def stem_and_check_stop(self,stopword_set):
        self.text = ' '.join([stemmer.stem(w) for w in self.text.split() if w not in stopword_set])

    def make_one_gram_tokens(self):
        self.one_gram_tokens = re.split(r'[-\s.,;!?]+', self.text)[:-1]


def print_sentences_by_clusters(clusters_dict, test_predict):
    for key in clusters_dict.keys():
        if key in test_predict:
            sentences_indexes_in_cluster = [index for index, value in enumerate(test_predict) if value == key]
            print(f'Test sentences in cluster number {key + 1}')
            for sentence_index in sentences_indexes_in_cluster:
                print(f'Sentence: {doc.test.get_original_sentences()[sentence_index]}')
            print('\n')
            print(f'Train sentences in cluster number {key + 1}')
            for index in clusters_dict[key]:
                print(doc.train.get_original_sentences()[index])
            print('\n')


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

for sentence in doc.sentences:
    sentence.stem_and_check_stop(stopword_set)
    sentence.make_one_gram_tokens()
    sentence.text=' '.join(sentence.one_gram_tokens)

#corpus = {}
#for i, sentence in enumerate(tokens):
#    corpus['sent' + str(i)] = Counter(sentence)
#df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

'''
#%% DBSCAN
epsilons=[0.1:0.1:1]
clustering = DBSCAN(eps=3, min_samples=50).fit(doc.train.tfidf.todense())
print("number of groups: " + str())
'''

#%% K-means

#train is in order to name the clusters
#validation is in order to choose the optimal k by comparing the group assignment by k means
#clusters_range=[30]
#for clusters_num in clusters_range:

#%% TF-IDF

doc.train_test_split(validation=50,test=50)
doc.train.make_lexicon()
tfidf_model = TfidfVectorizer(min_df=0.0, smooth_idf=True, norm='l1')
tfidf_trained=tfidf_model.fit(doc.train.get_sentences())
doc.train.make_tfidf(tfidf_trained)
doc.test.make_tfidf(tfidf_trained)



k = 30
kmeans_tfidf_model = KMeans(n_clusters=k, random_state=0).fit(doc.train.tfidf)
kmeans_tfidf_predict = kmeans_tfidf_model.predict(doc.test.tfidf)
tfidf_clusters_dict=doc.train.clusters_to_sentences_indexes_dict(kmeans_tfidf_model.labels_,k)

if __name__=="__main__":
    print(f'Clusters number = {k}\n')
    print_sentences_by_clusters(tfidf_clusters_dict,kmeans_tfidf_predict)


#%% Word2Vec
#if some words in our lexicon dont exist in words2vec lexicon change them to 'unknown' token
#check how to determine number of features

# Word2Vec Hyper-parameters:
# 1.Number of tokens
# 2. alpha - learning rate
# 3. window
# 4. Size of vector
# 5. Compare it to other models

# After the training one should load the saved model, so the following code should not executed again:

word2vec_model = Word2Vec(min_count=0,
                          window=5,
                          size=300,
                          sample=1e-3,
                          alpha=0.03,
                          min_alpha=0.0007,
                          workers=1)

train_tokens = doc.train.get_sentences_tokens()
word2vec_model.build_vocab(train_tokens)
word2vec_model.train(train_tokens,total_examples=word2vec_model.corpus_count,epochs=30)
#word2vec_model.save("word2vec.model")

word2vec_model = Word2Vec.load("word2vec.model")

for hyperparameter_lambda in [0,0.5,1]:

    doc.train.make_word2vec(word2vec_model,hyperparameter_lambda)
    doc.validation.make_word2vec(word2vec_model,hyperparameter_lambda)

    kmeans_word2vec_model = KMeans(n_clusters=k, random_state=0).fit(doc.train.word2vec[hyperparameter_lambda])
    kmeans_word2vec_predict = kmeans_word2vec_model.predict(doc.validation.word2vec[hyperparameter_lambda])
    word2vec_clusters_dict=doc.train.clusters_to_sentences_indexes_dict(kmeans_word2vec_model.labels_,k)

    if __name__ == "__main__":
        print(f'Clusters number = {k}, lambda = {hyperparameter_lambda}\n')
        print_sentences_by_clusters(word2vec_clusters_dict, kmeans_word2vec_predict)

'''
#### PubMed Word 2 Vec

word_vectors = KeyedVectors.load_word2vec_format('../RANZCR/PubMed-w2v.bin', binary=True)
#word_vectors.save("word2vec_pubmed.model")
word_vectors = Word2Vec.load("word2vec_pubmed.model")


sent_train = doc.train.get_original_sentences()

newl = list()
for i,v in enumerate(doc.train.get_original_sentences()):
    a = np.zeros(200)
    for j in v.split():
        if j not in word_vectors.vocab.keys():
            a += np.zeros(200)
        else:
            a += word_vectors.wv[j]
    newl.append(a)

newl_val = list()
for i,v in enumerate(doc.validation.get_original_sentences()):
    a = np.zeros(200)
    for j in v.split():
        if j not in word_vectors.vocab.keys():
            a += np.zeros(200)
        else:
            a += word_vectors.wv[j]
    newl_val.append(a)
'''

'''
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
'''

"""
#%% TF-IDF Manual
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