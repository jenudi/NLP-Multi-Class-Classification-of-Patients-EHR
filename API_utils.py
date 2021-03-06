from flask import Flask, request, jsonify
from classes import *
from RNN import *
from pymongo import MongoClient
from multiprocessing import Process, Value, Queue, cpu_count
from time import sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from math import inf
import torch
import pickle


args = NLP_args(k=30, min=0.0, random=0,min_cls=5,lr=0.0005)

labels_dict=pickle.load(open("labels_dict.pkl", "rb"))

#word2vec_for_kmeans_model=pickle.load(open("word2vec_for_kmeans_model.pkl", "rb"))
word2vec_for_kmeans_model = Word2Vec.load("word2vec_for_kmeans_model.model")



tfidf_model=pickle.load(open("tfidf_model.pkl", "rb"))

#word2vec_for_rnn_model=pickle.load(open("word2vec_for_rnn_model.pkl", "rb"))
word2vec_for_rnn_model = Word2Vec.load("word2vec_for_rnn_model.model")


rnn_model = RNN(args.word2vec_vec_size_for_rnn, args.hidden_layer, len(labels_dict))
rnn_model.load_state_dict(torch.load('w2v_5_rnn_model.pth'))
rnn_model.eval()

random_forest_model=pickle.load(open("random_forest_model.pkl", "rb"))

global number_of_free_processes
number_of_free_processes = Value('i', cpu_count(), lock=True)


client = MongoClient('mongodb://localhost:27017/')
with client:

    NLP_project_db = client["NLP_models_comparison"]
    sentences_collection = NLP_project_db["sentences"]
    tfidf_clusters_collection = NLP_project_db["tfidf_clusters"]
    word2vec_clusters_collection = NLP_project_db["word2vec_clusters"]

try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')
stopword_set = set(stopwords.words("english"))


def get_euclidiaan_distance(first_embedding, second_embedding):
    return np.square(np.sum(np.square(first_embedding - second_embedding)))


def find_closest_centroid(centroids_query,embedding):
    distances_list=list()
    for centroid_dict in centroids_query:
        centroid=centroid_dict["centroid"]
        euclidiaan_distance=get_euclidiaan_distance(centroid, embedding)
        distances_list.append((centroid,euclidiaan_distance))
    min_centroid_index=np.argmin([distance[1] for distance in distances_list])
    return distances_list[min_centroid_index][0]


class synchronizationError(Exception):
    pass


def check_number_of_free_processes():
    while True:
        if number_of_free_processes.value>=1:
            return
        else:
            sleep(1)


def increase_number_of_free_processes():
    number_of_free_processes.value += 1


def decrease_number_of_free_processes():
    number_of_free_processes.value -= 1
    if number_of_free_processes.value<0:
        raise synchronizationError("number_of_free_processes cannot be less than 0")


def is_valid_sentence(sentence):
    if (sentence is None or len(sentence) == 0) or (not isinstance(sentence, str)):
        return False
    else:
        return True