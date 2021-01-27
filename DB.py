from bson.son import SON
from pymongo import MongoClient
from main import args.doc, tfidf_centroids, word2vec_centroids, word2vec_pubmed_centroids, args.lambdas, args.k


if __name__ == "__main__":

    chosen_lambda=None
    while chosen_lambda not in args.lambdas:
        chosen_lambda=input("enter the chosen lambda from the following:" + str(args.lambdas))

    word2vec_models=["word2vec","word2vec_pubmed"]
    chosen_word2vec_model=None
    while chosen_word2vec_model not in word2vec_models:
        chosen_word2vec_model=input("choose one of the following models:"+ str(word2vec_models))
    if chosen_word2vec_model=="word2vec":
        chosen_word2vec_centroids=word2vec_centroids
    else:
        chosen_word2vec_centroids=word2vec_pubmed_centroids
        args.doc.train.word2vec_clusters_labels=args.doc.train.word2vec_pubmed_clusters_labels
        args.doc.validation.word2vec_clusters_labels=args.doc.validation.word2vec_pubmed_clusters_labels
        args.doc.test.word2vec_clusters_labels=args.doc.test.word2vec_pubmed_clusters_labels
        args.doc.train.word2vec=args.doc.train.word2vec_pubmed
        args.doc.validation.word2vec=args.doc.validation.word2vec_pubmed
        args.doc.test.word2vec=args.doc.test.word2vec_pubmed


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
                                            "centroid": list(map(float,chosen_word2vec_centroids[chosen_lambda][cluster_number]))
                                            }))

    sentence_id=0

    for set in [args.doc.train, args.doc.validation, args.doc.test]:
        for index,sentence in enumerate(set.sentences):

            sentence_id+=1
            sentence_tfidf_cluster=set.tfidf_clusters_labels[index]
            sentence_word2vec_cluster=set.word2vec_clusters_labels[chosen_lambda][index]

            tfidf_clusters_list[sentence_tfidf_cluster-1]["sentences in cluster"].append(sentence_id)
            word2vec_clusters_list[sentence_word2vec_cluster-1]["sentences in cluster"].append(sentence_id)

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