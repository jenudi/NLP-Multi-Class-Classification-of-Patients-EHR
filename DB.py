from bson.son import SON
from pymongo import MongoClient
from collections import Counter
from main import args, document, tfidf_centroids, word2vec_centroids


if __name__ == "__main__":

    sentences_list=list()
    tfidf_clusters_list=list()
    word2vec_clusters_list=list()

    for cluster_number in range(args.k):

        tfidf_clusters_list.append(SON({"_id": cluster_number+1,
                                        "sentences in cluster": list(),
                                        "centroid": list(map(float,tfidf_centroids[cluster_number])),
                                        "most common labels": list()
                                        }))

        word2vec_clusters_list.append(SON({"_id": cluster_number+1,
                                            "sentences in cluster": list(),
                                            "centroid": list(map(float,word2vec_centroids[cluster_number])),
                                            "most common labels": list()
                                           }))

    sentence_id=0

    for set in [document.train, document.validation, document.test]:
        for index,sentence in enumerate(set.sentences):

            sentence_id+=1
            sentence_tfidf_cluster=set.tfidf_clusters[index]
            sentence_word2vec_cluster=set.word2vec_clusters[index]

            tfidf_clusters_list[sentence_tfidf_cluster-1]["sentences in cluster"].append(sentence_id)
            tfidf_clusters_list[sentence_tfidf_cluster-1]["most common labels"].append(sentence.label)

            word2vec_clusters_list[sentence_word2vec_cluster-1]["sentences in cluster"].append(sentence_id)
            word2vec_clusters_list[sentence_tfidf_cluster-1]["most common labels"].append(sentence.label)


            new_sentence_document=SON({
                "_id": sentence_id,
                "text": sentence.original_text,
                "label": sentence.label,
                "tf-idf embeddings": list(map(float,set.tfidf[index].tolist()[:][0])),
                "word2vec embeddings": list(map(float,set.word2vec_for_kmeans[index].tolist())),
                })

            sentences_list.append(new_sentence_document)

    for cluster_list in [tfidf_clusters_list, word2vec_clusters_list]:
        for cluster_number in range(args.k):
            counter=Counter(cluster_list[cluster_number]["most common labels"])
            if len(counter.most_common(3))>2:
                cluster_list[cluster_number]["most common labels"]=[count[0] for count in counter.most_common(3)]
            elif len(counter.most_common(3))==2:
                cluster_list[cluster_number]["most common labels"]=[count[0] for count in counter.most_common(2)]
            else:
                cluster_list[cluster_number]["most common labels"] =[counter.most_common(1)[0][0]]


client = MongoClient('mongodb://localhost:27017/')
with client:

    NLP_project_db = client["NLP_models_comparison"]
    sentences_collection = NLP_project_db["sentences"]
    tfidf_clusters_collection = NLP_project_db["tfidf_clusters"]
    word2vec_clusters_collection = NLP_project_db["word2vec_clusters"]

    if __name__ == "__main__":
        sentences_collection.insert_many(sentences_list)
        tfidf_clusters_collection.insert_many(tfidf_clusters_list)
        word2vec_clusters_collection.insert_many(word2vec_clusters_list)