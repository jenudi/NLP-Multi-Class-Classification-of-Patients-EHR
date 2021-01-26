# %% Imports

from classes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import SnowballStemmer
from collections import Counter
import copy
from nltk.util import ngrams
import nltk


# %% Initialization
class ProArgs:

    def __init__(self, k=30, min=0.0, random=0, vec_size=300):
        self.k = k
        self.min = min
        self.random = random
        self.window = [3, 5]
        self.hyperp_lambda = [0, 0.5, 1]
        self.vec_size = vec_size
        self.doc = None


args = ProArgs()
args.doc = init_classes()


def print_sentences_by_clusters(args, clusters_dict, test_predict):
    for key in clusters_dict.keys():
        if key in test_predict:
            sentences_indexes_in_cluster = [index for index, value in enumerate(test_predict) if value == key]
            print(f'Test sentences in cluster number {key + 1}')
            for sentence_index in sentences_indexes_in_cluster:
                print(args.doc.test.get_original_sentences()[sentence_index])
            print('\n')
            print(f'Train sentences in cluster number {key + 1}')
            sentences_printed = 0
            for index in clusters_dict[key]:
                print(args.doc.train.get_original_sentences()[index])
                sentences_printed += 1
                if sentences_printed > 15:
                    break
            print('\n')


# %% TF-IDF
def tf_idf_model(args):
    tfidf_model = TfidfVectorizer(min_df=args.min,
                                  smooth_idf=True,
                                  norm='l1')

    tfidf_trained = tfidf_model.fit(args.doc.train.get_sentences())
    args.doc.train.make_tfidf(tfidf_trained)
    args.doc.validation.make_tfidf(tfidf_trained)
    args.doc.test.make_tfidf(tfidf_trained)

    kmeans_tfidf_model = KMeans(n_clusters=args.k,
                                random_state=args.random).fit(args.doc.train.tfidf)

    tfidf_centroids = kmeans_tfidf_model.cluster_centers_

    args.doc.train.tfidf_clusters_labels = kmeans_tfidf_model.labels_
    args.doc.validation.tfidf_clusters_labels = kmeans_tfidf_model.predict(args.doc.validation.tfidf)
    args.doc.test.tfidf_clusters_labels = kmeans_tfidf_model.predict(args.doc.test.tfidf)

    tfidf_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(args.doc.train.tfidf_clusters_labels,
                                                                            args.k)
    print(f'Clusters number = {args.k}\n')
    print_sentences_by_clusters(args,tfidf_clusters_dict,
                                args.doc.test.tfidf_clusters_labels)


# %%

if __name__ == "__main__":
    tf_idf_model(args)


# %% Word2Vec
def word2vec(args):
    train_tokens = args.doc.train.get_sentences_tokens()
    word2vec_centroids = dict()

    for window in args.window:
        word2vec_model = Word2Vec(min_count=args.min,
                                  window=window,
                                  size=args.vec_size,
                                  sample=1e-3,
                                  alpha=0.03,
                                  min_alpha=0.0007)

        word2vec_model.build_vocab(train_tokens)
        word2vec_model.train(train_tokens,
                             total_examples=word2vec_model.corpus_count,
                             epochs=30)

        for hyperp_lambda in args.hyperp_lambda:

            args.doc.train.make_word2vec(word2vec_model, hyperp_lambda, window)
            args.doc.validation.make_word2vec(word2vec_model, hyperp_lambda, window)
            args.doc.test.make_word2vec(word2vec_model, hyperp_lambda, window)

            kmeans_word2vec_model = KMeans(n_clusters=args.k,
                                           random_state=args.random).fit(
                                                                        args.doc.train.word2vec[(hyperp_lambda,
                                                                                                 window)])

            word2vec_centroids[(hyperp_lambda, window)] = kmeans_word2vec_model.cluster_centers_

            args.doc.train.word2vec_clusters_labels[(hyperp_lambda, window)] = kmeans_word2vec_model.labels_
            args.doc.validation.word2vec_clusters_labels[(hyperp_lambda, window)] = kmeans_word2vec_model.predict(
                args.doc.validation.word2vec[(hyperp_lambda, window)])
            args.doc.test.word2vec_clusters_labels[(hyperp_lambda, window)] = kmeans_word2vec_model.predict(
                args.doc.test.word2vec[(hyperp_lambda, window)])

            word2vec_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(
                args.doc.train.word2vec_clusters_labels[(hyperp_lambda, window)], args.k)
            print(f'Clusters number = {args.k}, lambda = {hyperp_lambda}, window size = {window} \n')
            print_sentences_by_clusters(args, word2vec_clusters_dict,
                                        args.doc.validation.word2vec_clusters_labels[(hyperp_lambda, window)])


#%%

if __name__ == "__main__":
    word2vec(args)


#%% PubMed Word2vec

def word2vec_pubmed(args):

    try:
        word2vec_pubmed_model = Word2Vec.load("word2vec_pubmed.model")
    except FileNotFoundError:
        print('No such model file')
        return

    word2vec_pubmed_centroids = dict()

    for hyperp_lambda in args.hyperp_lambda:

        args.doc.train.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)
        args.doc.validation.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)
        args.doc.test.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)

        kmeans_word2vec_pubmed_model = KMeans(n_clusters=args.k, random_state=args.random).fit(args.doc.train.word2vec_pubmed[hyperp_lambda])

        word2vec_pubmed_centroids[hyperp_lambda] = kmeans_word2vec_pubmed_model.cluster_centers_

        args.doc.train.word2vec_pubmed_clusters_labels[hyperp_lambda] = kmeans_word2vec_pubmed_model.labels_
        args.doc.validation.word2vec_pubmed_clusters_labels[hyperp_lambda] = kmeans_word2vec_pubmed_model.predict(
            args.doc.validation.word2vec_pubmed[hyperp_lambda])
        args.doc.test.word2vec_pubmed_clusters_labels[hyperp_lambda] = kmeans_word2vec_pubmed_model.predict(
            args.doc.test.word2vec_pubmed[hyperp_lambda])

        word2vec_pubmed_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(
            args.doc.train.word2vec_pubmed_clusters_labels[hyperp_lambda], args.k)
        print(f'Clusters number = {args.k}, lambda = {hyperp_lambda} \n')
        print_sentences_by_clusters(args,word2vec_pubmed_clusters_dict,
                                    args.doc.validation.word2vec_pubmed_clusters_labels[hyperp_lambda])


if __name__ == "__main__":
    word2vec_pubmed(args)
