import numpy as np
from collections import OrderedDict
from nltk.stem.porter import PorterStemmer
import torch
import re
from itertools import groupby
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
import pandas as pd


class NLP_args:

    def __init__(self, k=30, min=0.0, random=0, min_cls=5,
                 word2vec_vec_size_for_kmeans=300,
                 lr=0.0002, hidden_layer=200, epoch_num=15):
        self.k = k
        self.min = min
        self.random = random
        self.models = ['w2v_3', 'w2v_5', 'w2v_p']
        self.min_cls = min_cls
        self.lr = lr
        self.word2vec_vec_size_for_kmeans = word2vec_vec_size_for_kmeans
        self.hidden_layer = hidden_layer
        self.word2vec_vec_size_for_rnn = 300
        self.epoch_num = epoch_num
        self.l2 = 0.005


class Document:

    def __init__(self,text):
        self.text=text
        self.sentences=list()
        self.sentences_labels = list()
        self.train = Document_set(list())
        self.validation = Document_set(list())
        self.test = Document_set(list())
        self.labels_dict = dict()
        self.weights = list()
        self.cls_numbers = list()

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

    def make_dict(self):
        labels = [sentence.label for sentence in self.train.sentences]
        cls_w = np.array([len(list(group)) for key, group in groupby(labels)])
        cls_numbers = [list(dict.fromkeys(labels)).index(i) for i in labels]
        self.labels_dict = dict(zip( range(len(set(cls_numbers))), list(dict.fromkeys(labels) )))
        self.weights = torch.FloatTensor(1 - (cls_w / sum(cls_w)))
        self.cls_numbers = cls_numbers


class Document_set:

    def __init__(self, sentences):
        self.sentences = sentences
        self.lexicon = list()
        self.tfidf = list()
        self.tfidf_clusters=list()
        self.word2vec_for_kmeans = list()
        self.word2vec_clusters = dict()
        self.word2vec_for_rnn=list()
        self.word2vec_model_name = None

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

    def make_word2vec_for_kmeans(self,word2vec_model, vector_size):
        for sentence_tokens in self.get_sentences_tokens():
            word_embedding = np.mean([word2vec_model.wv[token] if token in word2vec_model.wv.vocab.keys()
                                   else np.zeros(vector_size) for token in sentence_tokens],axis=0)
            self.word2vec_for_kmeans.append(word_embedding /np.linalg.norm(word_embedding))

    def make_word2vec_for_rnn(self,args, window=None):
        train_tokens = self.get_sentences_tokens()
        if window is not None:
            self.word2vec_model_name = f'w2v_{window}'
            word2vec_model = Word2Vec(min_count=args.min, window=window, size=300,
                                      sample=1e-3, alpha=0.03, min_alpha=0.0007)
            word2vec_model.build_vocab(train_tokens)
            word2vec_model.train(train_tokens, total_examples=word2vec_model.corpus_count, epochs=30)
        else:
            try:
                self.word2vec_model_name = 'w2v_p'
                word2vec_model = KeyedVectors.load_word2vec_format('PubMed-w2v.bin', binary=True)
                # word2vec_pubmed_model = Word2Vec.load("word2vec_pubmed.model")
            except FileNotFoundError:
                print('No such model file')
                return
        self.word2vec_for_rnn = word2vec_model

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

    def preprocess_sentence_for_API(sentence, stopword_set):
        sentence.do_replaces()
        sentence.stem_and_check_stop(stopword_set)
        sentence.make_tokens()
        sentence.make_original_text_tokens()
        sentence.text = ' '.join(sentence.tokens)

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