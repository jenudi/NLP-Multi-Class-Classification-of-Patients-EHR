from flask import Flask, request, jsonify
from classes import *
from RNN import *
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from math import inf
import torch
import pickle


args = NLP_args(k=30, min=0.0, random=0, hidden=350,min_cls=5, lr=0.0005)

labels_dict=pickle.load(open("labels_dict.pkl", "rb"))

word2vec_for_kmeans_model=pickle.load(open("word2vec_for_kmeans_model.pkl", "rb"))

tfidf_model=pickle.load(open("tfidf_model.pkl", "rb"))

word2vec_for_rnn_model=pickle.load(open("word2vec_for_rnn_model.pkl", "rb"))

rnn_model = RNN(args.word2vec_vec_size_for_rnn, args.hidden, len(labels_dict))
rnn_model.load_state_dict(torch.load('rnn_model.pth'))
rnn_model.eval()

random_forest_model=pickle.load(open("random_forest_model.pkl", "rb"))


client = MongoClient('mongodb://localhost:27017/')
with client:

    NLP_project_db = client["NLP_models_comparison_project"]
    sentences_collection = NLP_project_db["sentences"]
    tfidf_clusters_collection = NLP_project_db["tfidf_clusters"]
    word2vec_clusters_collection = NLP_project_db["word2vec_clusters"]


try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')
stopword_set = set(stopwords.words("english"))


def get_euclidiaan_distance(first_embeddings, second_embeddings):
    return np.square(np.sum(np.square(first_embeddings - second_embeddings)))


def find_closest_centroid(centroids_query,embeddings):
    distances_list=list()
    for centroid_dict in centroids_query:
        centroid=centroid_dict["centroid"]
        euclidiaan_distance=get_euclidiaan_distance(centroid, embeddings)
        distances_list.append((centroid,euclidiaan_distance))
    min_centroid_index=np.argmin([distance[1] for distance in distances_list])
    return distances_list[min_centroid_index][0]