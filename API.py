from API_utils import *


app = Flask(__name__)


@app.route("/word2vec_cluster",methods=["POST"])
def word2vec_cluster():

    sentence=eval(request.get_json())["sentence"]
    if sentence is None or len(sentence)==0:
        return jsonify({"error":"no sentence"}), 400
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"}), 400

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        word2vec_embeddings = np.mean([word2vec_for_kmeans_model.wv[token] if token in word2vec_for_kmeans_model.wv.vocab.keys()
                                    else np.zeros(args.word2vec_vec_size_for_kmeans) for token in sentence_object.tokens], axis=0)

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
        return jsonify({"most common labels in cluster":cluster_labels, "closest sentences in cluster":closest_sentences_texts}), 200

    except:
        return jsonify({"error":"model failed"}), 500


@app.route("/tfidf_cluster",methods=["POST"])
def tfidf_cluster():

    sentence=eval(request.get_json())["sentence"]
    if sentence is None or len(sentence)==0:
        return jsonify({"error":"no sentence"}), 400
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"}), 400

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        tfidf_embeddings=tfidf_model.transform([sentence_object.text]).todense()

        centroids_query = tfidf_clusters_collection.find({}, {"centroid": 1, "_id": 0})
        closest_centroid=find_closest_centroid(centroids_query,tfidf_embeddings)
        cluster_query=tfidf_clusters_collection.find({"centroid":closest_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

        cluster_sentences_ids = cluster_query[0]["sentences in cluster"]
        closest_sentences_distances = [inf, inf, inf, inf, inf]
        closest_sentences_texts = ["", "", "", "", ""]
        for sentence_id in cluster_sentences_ids:
            sentence_query = sentences_collection.find({"_id": sentence_id})
            sentence_embeddings = sentence_query[0]["tf-idf embeddings"]
            euclidiaan_distance = get_euclidiaan_distance(tfidf_embeddings, sentence_embeddings)
            if euclidiaan_distance < max(closest_sentences_distances):
                change_index = closest_sentences_distances.index(max(closest_sentences_distances))
                closest_sentences_distances[change_index] = euclidiaan_distance
                closest_sentences_texts[change_index] = sentence_query[0]["text"]

        cluster_labels = cluster_query[0]["most common labels"]
        return jsonify({"most common labels in cluster": cluster_labels, "closest sentences in cluster": closest_sentences_texts}), 200

    except:
        return jsonify({"error":"model failed"}), 500



@app.route("/rnn_classification",methods=["POST"])
def word2vec_rnn_classification():

    sentence=eval(request.get_json())["sentence"]
    if sentence is None or len(sentence)==0:
        return jsonify({"error":"no sentence"}), 400
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"}), 400

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        input_tensor = torch.zeros(len(sentence_object.tokens), 1, args.word2vec_vec_size_for_rnn)
        for index, token in enumerate(sentence_object.tokens):
            numpy_copy = word2vec_for_rnn_model.wv[token].copy()
            input_tensor[index][0][:] = torch.from_numpy(numpy_copy)
        with torch.no_grad():
            rnn_model.eval()
            hidden = rnn_model.init_hidden()
            for i in range(input_tensor.size()[0]):
                output, hidden = rnn_model(input_tensor[i], hidden)
            predicted_label_number = int(torch.max(output, 1)[1].detach())
    
        predicted_label = labels_dict[predicted_label_number]
        return jsonify({"predicted label": predicted_label}), 200
    
    except:
        return jsonify({"error":"model failed"}), 500


@app.route("/random_forest_classification",methods=["POST"])
def tfidf_random_forest_classification():

    sentence=eval(request.get_json())["sentence"]
    if sentence is None or len(sentence)==0:
        return jsonify({"error":"no sentence"}), 400
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"}), 400

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        tfidf_embeddings=tfidf_model.transform([sentence_object.text]).todense()
        predicted_label=random_forest_model.predict(tfidf_embeddings)
        return jsonify({"predicted label":predicted_label[0]}), 200

    except:
        return jsonify({"error":"model failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)