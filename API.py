from flask import Flask, request, jsonify
from classes import *
from DB import sentences_collection, tfidf_clusters_collection, word2vec_clusters_collection
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from main import word2vec_chosen_model, word2vec_chosen_vector_size, tfidf_model, random_forest_chosen_model


try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')
stopword_set = set(stopwords.words("english"))


def find_closect_centroid(centroids_query,embeddings):
    distances_list=list()
    for centroid_dict in centroids_query:
        centroid=centroid_dict["centroid"]
        euc_distance=np.square(np.sum(np.square(centroid - embeddings)))
        distances_list.append((centroid,euc_distance))
    min_centroid_index=np.argmin([distance[1] for distance in distances_list])
    return distances_list[min_centroid_index][0]


app = Flask(__name__)


@app.route("/word2vec_cluster/<str:sentence>")
def word2vec_cluster(sentence,word2vec_model,vector_size,stopword_set):

    if sentence is None:
        return jsonify({"error":"no sentence"})
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    word_embeddings = np.mean([word2vec_model.wv[word] if word in word2vec_model.wv.vocab.keys()
                                else np.zeros(vector_size) for word in sentence_object.text], axis=0)
    normalized_embeddings= word_embeddings / np.linalg.norm(word_embeddings)

    centroids_query = word2vec_clusters_collection.find({}, {"centroid": 1, "_id": 0})
    closect_centroid=find_closect_centroid(centroids_query,normalized_embeddings)
    cluster_query=word2vec_clusters_collection.find({"centroid":closect_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

    returned_cluster_labels=cluster_query[0]["most_common_labels"]
    returned_sentences_indexes=np.random.randint(len(cluster_query[0]["sentences in cluster"]),size=(1,10))

    returned_sentences_ids = [cluster_query[0]["sentences in cluster"][index] for index in returned_sentences_indexes]


@app.route("/tfidf_cluster/<str:sentence>")
def tfidf_cluster(sentence,tfidf_model,stopword_set):

    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    tfidf_embeddings=tfidf_model.transform(sentence_object.text).todense()

    centroids_query = tfidf_clusters_collection.find({}, {"centroid": 1, "_id": 0})
    closect_centroid=find_closect_centroid(centroids_query,tfidf_embeddings)
    cluster_query=tfidf_clusters_collection.find({"centroid":closect_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

    returned_cluster_labels=cluster_query[0]["most_common_labels"]
    returned_sentences_indexes=np.random.randint(len(cluster_query[0]["sentences in cluster"]),size=(1,10))

    returned_sentences_ids = [cluster_query[0]["sentences in cluster"][index] for index in returned_sentences_indexes]


@app.route("/word2vec_rnn_classification/<str:sentence>")
def word2vec_rnn_classification(sentence):
    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)


@app.route("/tfidf_random_forest_classification/<str:sentence>")
def tfidf_random_forest_classification(sentence):
    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)


if __name__ == "__main__":
    app.run(host="127.0.0.1.", port=8000)