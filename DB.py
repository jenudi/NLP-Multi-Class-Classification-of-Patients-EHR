from bson.son import SON
from pymongo import MongoClient
from main import doc,chosen_lambda

if __name__ == "__main__":

    sentences_collection_list=list()
    tfidf_clusters_list=list()
    word2vec_clusters_list=list()

    for set in [doc.train, doc.validation, doc.test]:
        for index,sentence in enumerate(set.sentences):

            new_sentence_document=SON({
                "original text": sentence.original_text,
                "tf-idf representetion": list(map(float,set.tfidf[index].tolist()[:][0])),
                "tf-idf cluster": int(set.tfidf_clusters_labels[index]),
                "word2vec representetion": list(map(float,set.word2vec[chosen_lambda][index].tolist())),
                "word2vec cluster": int(set.word2vec_clusters_labels[chosen_lambda][index])
            })

            sentences_collection_list.append(new_sentence_document)

    client = MongoClient('mongodb://localhost:27017/')
    with client:

        NLP_models_comparison_project_db = client["NLP_models_comparison_project"]

        sentences_collection = NLP_models_comparison_project_db["sentences"]
        sentences_collection.insert_many(sentences_collection_list)

        tfidf_clusters_collection = NLP_models_comparison_project_db["tfidf_clusters"]
        tfidf_clusters_collection.insert_many(tfidf_clusters_list)

        tfidf_clusters_collection = NLP_models_comparison_project_db["word2vec_clusters"]
        tfidf_clusters_collection.insert_many(word2vec_clusters_list)