from bson.son import SON
from pymongo import MongoClient
from main import doc,chosen_lambda, k, tfidf_centroids, word2vec_centroids

if __name__ == "__main__":

    sentences_list=list()
    tfidf_clusters_list=list()
    word2vec_clusters_list=list()

    for cluster_number in range(k):

        tfidf_clusters_list.append(SON({"cluster number": cluster_number+1,
                                   "sentences ids":list(),
                                   "centroid":tfidf_centroids[cluster_number]
                                   }))

        word2vec_clusters_list.append(SON({"cluster number": cluster_number+1,
                                      "sentences ids": list(),
                                      "centroid": tfidf_centroids[cluster_number]
                                      }))

    sentence_id=0

    for set in [doc.train, doc.validation, doc.test]:
        for index,sentence in enumerate(set.sentences):

            sentence_id+=1

            tfidf_clusters_list[set.tfidf_clusters_labels[index]-1]["sentences ids"].append(int(set.tfidf_clusters_labels[index]))
            word2vec_clusters_list[set.word2vec_clusters_labels[chosen_lambda][index]-1]["sentences ids"].append(int(set.word2vec_clusters_labels[chosen_lambda][index]))

            new_sentence_document=SON({
                "_id": sentence_id,
                "original text": sentence.original_text,
                "tf-idf representetion": list(map(float,set.tfidf[index].tolist()[:][0])),
                "word2vec representetion": list(map(float,set.word2vec[chosen_lambda][index].tolist())),
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