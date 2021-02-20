import numpy as np
import re
import random
from collections import OrderedDict
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import torch

class NLPargs:

    def __init__(self, k=30, min=0.0, random=0, vec_size=300, hidden=128,min_cls=5):
        self.k = k
        self.min = min
        self.random = random
        self.windows = [3, 5]
        self.hyperp_lambdas = [0, 0.5, 1]
        self.vec_size = vec_size
        self.hidden = hidden
        self.doc = None
        self.min_cls = min_cls
        self.lr = 0.005


class Document:

    def __init__(self,text):
        self.text=text
        self.sentences=list()
        self.sentences_labels = list()
        self.train = Document_set(list())
        self.validation = Document_set(list())
        self.test = Document_set(list())

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
        self.sentences = [Sentence_in_document(sentence) for sentence in self.text.split(char)]

    def get_sentences(self):
        return [sentence.text for sentence in self.sentences]

    def train_test_split(self):
        for index,sentence in enumerate(self.sentences):
            if index % 9 == 0:
                self.validation.sentences.append(sentence)
            elif index % 10 == 0:
                self.test.sentences.append(sentence)
            else:
                self.train.sentences.append(sentence)


class Document_set:

    def __init__(self, sentences):
        self.sentences = sentences
        self.lexicon = list()
        self.tfidf = list()
        self.tfidf_clusters=list()
        self.word2vec = dict()
        self.word2vec_clusters = dict()
        self.word2vec_pubmed = dict()
        self.word2vec_pubmed_clusters = dict()
        self.labels_dict = dict()
        self.weights = list()

    def make_labels_dict_and_weights(self):
        temp_list = [i.label for i in self.sentences]
        cls_w = data['cc'].value_counts().tolist()
        cls_numbers = [list(dict.fromkeys(temp_list)).index(i) for i in temp_list]
        self.labels_dict = dict(zip( range(len(set(cls_numbers)))  , list(dict.fromkeys(temp_list))   ))
        self.weights = torch.FloatTensor([1 - (x / sum(cls_w)) for x in cls_w][::-1])

    def get_sentences(self):
        return [sentence.text for sentence in self.sentences]

    def get_original_sentences(self):
        return [sentence.original_text for sentence in self.sentences]

    def get_sentences_tokens(self):
        return [sentence.tokens for sentence in self.sentences]

    def get_original_text_sentences_tokens(self):
        return [sentence.original_text_tokens for sentence in self.sentences]

    def make_lexicon(self):
        doc_tokens=list()
        for sentence in self.sentences:
            doc_tokens += [sorted(sentence.tokens)]
        self.lexicon = sorted(set(sum(doc_tokens, [])))

    def get_zero_vector(self):
        return OrderedDict((tok, 0) for tok in self.lexicon)

    def make_tfidf(self,model):
        self.tfidf = model.transform(self.get_sentences()).todense()

    def make_word2vec(self,word2vec_model,hyperparameter_lambda,hyperparameter_window_size):
        self.word2vec[(hyperparameter_lambda,hyperparameter_window_size)]=list()
        for sentence_tokens in self.get_sentences_tokens():
            word_embeddings = np.mean([word2vec_model.wv[token] if token in word2vec_model.wv.vocab.keys()
                                   else np.zeros(300) for token in sentence_tokens],axis=0)
            word_embeddings *= (len(sentence_tokens) ** hyperparameter_lambda)
            self.word2vec[(hyperparameter_lambda, hyperparameter_window_size)].append(word_embeddings /
                                                                                      np.linalg.norm(word_embeddings))
    def make_word2vec_pubmed(self,word2vec_pubmed_model,hyperparameter_lambda):
        self.word2vec_pubmed[hyperparameter_lambda]=list()
        for sentence_tokens in self.get_original_text_sentences_tokens():
            word_embeddings = np.mean([word2vec_pubmed_model.wv[token] if token in word2vec_pubmed_model.wv.vocab.keys()
                                       else np.zeros(200) for token in sentence_tokens], axis=0)
            word_embeddings *= (len(sentence_tokens) ** hyperparameter_lambda)
            self.word2vec_pubmed[hyperparameter_lambda].append(word_embeddings / np.linalg.norm(word_embeddings))

    def clusters_to_sentences_indexes_dict(self,clusters,num_of_clusters):
        clusters_sentences_indexes_dict=dict()
        for cluster_num in range(num_of_clusters):
            true_clusters = clusters==cluster_num
            clusters_sentences_indexes_dict[cluster_num]=[index for index, truth_value in enumerate(true_clusters) if truth_value]
        return clusters_sentences_indexes_dict


class Sentence_in_document:
    def __init__(self,text):
        self.text=text
        self.original_text=text
        self.label=None
        self.tokens=list()
        self.original_text_tokens=list()

    def stem_and_check_stop(self,stopword_set):
        stemmer = PorterStemmer()
        # stemmer = SnowballStemmer('english')
        self.text = ' '.join([stemmer.stem(w) for w in self.text.split() if w not in stopword_set])

    def make_tokens(self):
        self.tokens = re.split(r'[-\s.,;!?]+', self.text)[:-1]

    def make_original_text_tokens(self):
        self.original_text_tokens = re.split(r'[-\s.,;!?]+', self.original_text)[:-1]

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