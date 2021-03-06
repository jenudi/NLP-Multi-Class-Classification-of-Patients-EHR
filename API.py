from API_utils import *



app = Flask(__name__)



def handle_request(sentence,function_name):
    check_number_of_free_processes()
    queue=Queue()
    decrease_number_of_free_processes()
    new_process=Process(target=function_name,args=(sentence,queue))
    new_process.start()
    new_process.join()
    increase_number_of_free_processes()
    response=queue.get()
    queue.close()
    return response



@app.route("/word2vec_cluster",methods=["POST"])
def word2vec_cluster():
    sentence=eval(request.get_json())["sentence"]
    response=handle_request(sentence,word2vec_cluster_function)
    return jsonify(response[0]),response[1]



@app.route("/tfidf_cluster",methods=["POST"])
def tfidf_cluster():
    sentence=eval(request.get_json())["sentence"]
    response=handle_request(sentence,tfidf_cluster_function)
    return jsonify(response[0]),response[1]



@app.route("/word2vec_rnn_classification",methods=["POST"])
def word2vec_rnn_classification():
    sentence=eval(request.get_json())["sentence"]
    response=handle_request(sentence,word2vec_rnn_classification_function)
    return jsonify(response[0]),response[1]



@app.route("/tfidf_random_forest_classification",methods=["POST"])
def tfidf_random_forest_classification():
    sentence=eval(request.get_json())["sentence"]
    response=handle_request(sentence,tfidf_random_forest_classification_function)
    return jsonify(response[0]),response[1]



def word2vec_cluster_function(sentence,queue):
    if not is_valid_sentence(sentence):
        queue.put([{"error":"sentence invalid format"}, 400])
        return

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        word2vec_embedding = np.mean([word2vec_for_kmeans_model.wv[token] if token in word2vec_for_kmeans_model.wv.vocab.keys()
                                    else np.zeros(args.word2vec_vec_size_for_kmeans) for token in sentence_object.tokens], axis=0)

        normalized_embedding= word2vec_embedding / np.linalg.norm(word2vec_embedding)

        centroids_query = word2vec_clusters_collection.find({}, {"centroid": 1, "_id": 0})
        closest_centroid=find_closest_centroid(centroids_query,normalized_embedding)
        cluster_query = word2vec_clusters_collection.find({"centroid":closest_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

        cluster_sentences_ids = cluster_query[0]["sentences in cluster"]
        closest_sentences_distances = [inf, inf, inf, inf, inf]
        closest_sentences_texts=["", "", "", "", ""]
        for sentence_id in cluster_sentences_ids:
            sentence_query = sentences_collection.find({"_id":sentence_id},{"word2vec embedding":1, "text":1, "_id":0})
            sentence_embedding=sentence_query[0]["word2vec embedding"]
            euclidiaan_distance=get_euclidiaan_distance(normalized_embedding, sentence_embedding)
            if euclidiaan_distance < max(closest_sentences_distances):
                change_index=closest_sentences_distances.index(max(closest_sentences_distances))
                closest_sentences_distances[change_index]=euclidiaan_distance
                closest_sentences_texts[change_index]=sentence_query[0]["text"]

        cluster_labels=cluster_query[0]["most common labels"]
        queue.put([{"most common labels in cluster":cluster_labels, "closest sentences in cluster":closest_sentences_texts}, 200])
        return

    except:
        queue.put([{"error":"model failed"}, 500])
        return



def tfidf_cluster_function(sentence,queue):
    if not is_valid_sentence(sentence):
        queue.put([{"error":"sentence invalid format"}, 400])
        return

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        tfidf_embedding=tfidf_model.transform([sentence_object.text]).todense()
        centroids_query = tfidf_clusters_collection.find({}, {"centroid": 1, "_id": 0})
        closest_centroid=find_closest_centroid(centroids_query,tfidf_embedding)
        cluster_query=tfidf_clusters_collection.find({"centroid":closest_centroid}, {"sentences in cluster": 1, "most common labels": 1, "_id": 0})

        cluster_sentences_ids = cluster_query[0]["sentences in cluster"]
        closest_sentences_distances = [inf, inf, inf, inf, inf]
        closest_sentences_texts = ["", "", "", "", ""]
        for sentence_id in cluster_sentences_ids:
            sentence_query = sentences_collection.find({"_id": sentence_id})
            sentence_embedding = sentence_query[0]["tf-idf embedding"]
            euclidiaan_distance = get_euclidiaan_distance(tfidf_embedding, sentence_embedding)
            if euclidiaan_distance < max(closest_sentences_distances):
                change_index = closest_sentences_distances.index(max(closest_sentences_distances))
                closest_sentences_distances[change_index] = euclidiaan_distance
                closest_sentences_texts[change_index] = sentence_query[0]["text"]

        cluster_labels = cluster_query[0]["most common labels"]
        queue.put([{"most common labels in cluster": cluster_labels, "closest sentences in cluster": closest_sentences_texts}, 200])
        return

    except:
        queue.put([{"error":"model failed"}, 500])
        return


def word2vec_rnn_classification_function(sentence,queue):
    if not is_valid_sentence(sentence):
        queue.put([{"error":"sentence invalid format"}, 400])
        return

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    #try:
    input_tensor = torch.zeros(len(sentence_object.tokens), 1, args.word2vec_vec_size_for_rnn)
    for index, token in enumerate(sentence_object.tokens):
        try:
            numpy_copy = word2vec_for_rnn_model.wv[token].copy()
        except KeyError:
            numpy_copy = np.zeros(args.word2vec_vec_size_for_rnn)
        input_tensor[index][0][:] = torch.from_numpy(numpy_copy)
    with torch.no_grad():
        hidden = rnn_model.init_hidden()
        for i in range(input_tensor.size()[0]):
            output, hidden = rnn_model(input_tensor[i], hidden)
        predicted_label_number = int(torch.max(output, 1)[1].detach())

    predicted_label = labels_dict[predicted_label_number]
    queue.put([{"predicted label": predicted_label}, 200])
    return
    '''
    except:
        queue.put([{"error":"model failed"}, 500])
        return
'''


def tfidf_random_forest_classification_function(sentence,queue):
    if not is_valid_sentence(sentence):
        queue.put([{"error":"sentence invalid format"}, 400])
        return

    sentence_object = Sentence_in_document(sentence.strip().lower())
    sentence_object.preprocess_sentence_for_API(stopword_set)

    try:
        tfidf_embedding=tfidf_model.transform([sentence_object.text]).todense()
        predicted_label=random_forest_model.predict(tfidf_embedding)
        queue.put([{"predicted label":predicted_label[0]}, 200])
        return

    except:
        queue.put([{"error":"model failed"}, 500])
        return



if __name__ == "__main__":
    app.run(debug=True)