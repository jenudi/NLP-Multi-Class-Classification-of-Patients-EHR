from flask import Flask, request, jsonify
from classes import *
from DB import sentences_collection, tfidf_clusters_collection, word2vec_clusters_collection
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from math import inf
import torch
import pickle


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