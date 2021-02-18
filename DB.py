from bson.son import SON
from pymongo import MongoClient
from main import args, tfidf_centroids, word2vec_centroids,word2vec_chosen_params


if __name__ == "__main__":

    sentences_list=list()
    tfidf_clusters_list=list()
    word2vec_clusters_list=list()

    for cluster_number in range(args.k):

        tfidf_clusters_list.append(SON({"_id": cluster_number+1,
                                        "sentences in cluster": list(),
                                        "centroid": list(map(float,tfidf_centroids[cluster_number]))
                                        }))

        word2vec_clusters_list.append(SON({"_id": cluster_number+1,
                                            "sentences in cluster": list(),
                                            "centroid": list(map(float,word2vec_centroids[word2vec_chosen_params][cluster_number]))
                                            }))

    sentence_id=0

    for set in [args.doc.train, args.doc.validation, args.doc.test]:
        for index,sentence in enumerate(set.sentences):

            sentence_id+=1
            sentence_tfidf_cluster=set.tfidf_clusters[index]
            sentence_word2vec_cluster=set.word2vec_clusters[word2vec_chosen_params][index]

            tfidf_clusters_list[sentence_tfidf_cluster-1]["sentences in cluster"].append(sentence_id)
            word2vec_clusters_list[sentence_word2vec_cluster-1]["sentences in cluster"].append(sentence_id)

            new_sentence_document=SON({
                "_id": sentence_id,
                "original text": sentence.original_text,
                "label": sentence.label,
                "tf-idf representetion": list(map(float,set.tfidf[index].tolist()[:][0])),
                "word2vec representetion": list(map(float,set.word2vec[word2vec_chosen_params][index].tolist())),
                })

            sentences_list.append(new_sentence_document)

    client = MongoClient('mongodb://localhost:27017/')
    with client:

        NLP_models_comparison_project_db = client["NLP_models_comparison_project"]

        sentences_collection = NLP_models_comparison_project_db["sentences"]
        sentences_collection.insert_many(sentences_list)

        tfidf_clusters_collection = NLP_models_comparison_project_db["tfidf_clusters"]
        tfidf_clusters_collection.insert_many(tfidf_clusters_list)

        word2vec_clusters_collection = NLP_models_comparison_project_db["word2vec_clusters"]
        word2vec_clusters_collection.insert_many(word2vec_clusters_list)