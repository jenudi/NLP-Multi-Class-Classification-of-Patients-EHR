from flask import Flask, request, jsonify
from classes import *
from DB import sentences_collection, tfidf_clusters_collection, word2vec_clusters_collection
from main import word2vec_for_kmeans_model , tfidf_model, random_forest_chosen_model, document, args
#from main word2vec_for_rnn_chosen_vector_size, rnn_model
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from math import inf
import torch


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


app = Flask(__name__)


@app.route("/word2vec_cluster/<str:sentence>")
def word2vec_cluster(sentence ,stopword_set):

    if sentence is None:
        return jsonify({"error":"no sentence"})
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    word2vec_embeddings = np.mean([word2vec_for_kmeans_model.wv[word] if word in word2vec_for_kmeans_model.wv.vocab.keys()
                                else np.zeros(args.args.word2vec_vec_size_for_kmeans) for word in sentence_object.text], axis=0)
    normalized_embeddings= word2vec_embeddings / np.linalg.norm(word2vec_embeddings)

    centroids_query = word2vec_clusters_collection.find({}, {"centroid": 1, "_id": 0})
    closest_centroid=find_closest_centroid(centroids_query,normalized_embeddings)
    cluster_query=word2vec_clusters_collection.find({"centroid":closest_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

    cluster_sentences_ids = cluster_query[0]["sentences in cluster"]
    closest_sentences_distances = [inf, inf, inf, inf, inf]
    closest_sentences_texts=["", "", "", "", ""]
    for sentence_id in cluster_sentences_ids:
        sentence_query=sentences_collection.find({"_id":sentence_id},{"word2vec embeddings":1, "text":1, "_id":0})
        sentence_embeddings=sentence_query[0]["word2vec embeddings"]
        euclidiaan_distance=get_euclidiaan_distance(normalized_embeddings, sentence_embeddings)
        if euclidiaan_distance < max(closest_sentences_distances):
            change_index=closest_sentences_distances.index(max(closest_sentences_distances))
            closest_sentences_distances[change_index]=euclidiaan_distance
            closest_sentences_texts[change_index]=sentence_query[0]["text"]

    cluster_labels=cluster_query[0]["most common labels"]
    return jsonify({"most common labels in cluster":cluster_labels, "closest sentences in cluster":closest_sentences_texts})


@app.route("/tfidf_cluster/<str:sentence>")
def tfidf_cluster(sentence,stopword_set):

    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    tfidf_embeddings=tfidf_model.transform(sentence_object.text).todense()

    centroids_query = tfidf_clusters_collection.find({}, {"centroid": 1, "_id": 0})
    closest_centroid=find_closest_centroid(centroids_query,tfidf_embeddings)
    cluster_query=tfidf_clusters_collection.find({"centroid":closest_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

    cluster_sentences_ids = cluster_query[0]["sentences in cluster"]
    closest_sentences_distances = [inf, inf, inf, inf, inf]
    closest_sentences_texts = ["", "", "", "", ""]
    for sentence_id in cluster_sentences_ids:
        sentence_query = sentences_collection.find({"_id": sentence_id},{"tf-idf embeddings:1, text:1, _id:0"})
        sentence_embeddings = sentence_query[0]["tf-idf embeddings"]
        euclidiaan_distance = get_euclidiaan_distance(tfidf_embeddings, sentence_embeddings)
        if euclidiaan_distance < max(closest_sentences_distances):
            change_index = closest_sentences_distances.index(max(closest_sentences_distances))
            closest_sentences_distances[change_index] = euclidiaan_distance
            closest_sentences_texts[change_index] = sentence_query[0]["text"]

    cluster_labels = cluster_query[0]["most common labels"]
    return jsonify({"most common labels in cluster": cluster_labels, "closest sentences in cluster": closest_sentences_texts})


@app.route("/rnn_classification/<str:sentence>")
def word2vec_rnn_classification(sentence):
    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    '''        
    input_tensor = torch.zeros(len(sentence_object.tokens), 1, args.word2vec_vec_size_for_kmeans)
    for index, token in enumerate(tokens):
        numpy_copy = document.train.word2vec_for_rnn.wv[token].copy()
        input_tensor[index][0][:] = torch.from_numpy(numpy_copy)
    with torch.no_grad():
        rnn_model.eval()
        hidden = rnn_model.init_hidden()
        for i in range(input_tensor.size()[0]):
            output, hidden = rnn_model(input_tensor[i], hidden)
        predicted_label_number = int(torch.max(output, 1)[1].detach())
        
    predicted_label=document.train.labels_dict[predicted_label_number]
    return jsonify({"predicted label":predicted_label})
    '''


@app.route("/random_forest_classification/<str:sentence>")
def tfidf_random_forest_classification(sentence):
    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    tfidf_embeddings=tfidf_model.transform(sentence_object.text).todense()

    predicted_label=random_forest_chosen_model.predict(tfidf_embeddings)
    return jsonify({"predicted label":predicted_label})


if __name__ == "__main__":
    app.run(host="127.0.0.1.", port=8000)