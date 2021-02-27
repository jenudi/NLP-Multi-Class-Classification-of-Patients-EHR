from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from classes import *


def make_tsne(model, model_name, labels):
    fig = plt.figure()
    tsne = TSNE()
    palette = sns.color_palette("icefire", len(set(labels)))
    model_tsne_embedded = tsne.fit_transform(model)
    temp_df = pd.DataFrame({'x1':model_tsne_embedded[:, 0],'x2':model_tsne_embedded[:, 1],'y':labels})
    centers = temp_df.groupby(by=["y"]).mean()
    sns.scatterplot(x=model_tsne_embedded[:, 0], y=model_tsne_embedded[:, 1], legend='full', palette=palette, hue=labels)
    plt.scatter(centers.iloc[:,0].values, centers.iloc[:,1].values, c=palette, s=200, alpha=0.5)
    fig.suptitle(f'Model: {model_name}', fontsize=16)
    plt.show()


def print_sentences_by_clusters(document, clusters_dict, validation_predict):
    for key in clusters_dict.keys():
        if key in validation_predict:
            validation_sentences_indexes_in_cluster = [index for index, value in enumerate(validation_predict) if
                                                       value == key]
            print(
                f'Validation sentences in cluster number {key + 1}, number of sentences: {len(validation_sentences_indexes_in_cluster)}\n')
            for sentence_index in validation_sentences_indexes_in_cluster:
                print(document.test.get_original_sentences()[sentence_index])
        else:
            print(f'No validation sentences in cluster number {key + 1}\n')
        print('\n')
        print(f'Train sentences in cluster number {key + 1}, size of cluster: {len(clusters_dict[key])} \n')
        sentences_printed = 0
        for index in clusters_dict[key]:
            print(document.train.get_original_sentences()[index])
            sentences_printed += 1
            if sentences_printed > 15:
                break
        print('\n')
        print('\n')


def word2vec_kmeans(document,args,word2vec_model, vector_size,print_sentences=False,t_sne=False):

    document.train.make_word2vec_for_kmeans(word2vec_model, vector_size)
    document.validation.make_word2vec_for_kmeans(word2vec_model, vector_size)
    document.test.make_word2vec_for_kmeans(word2vec_model, vector_size)

    kmeans_model = KMeans(n_clusters=args.k,
                                   random_state=args.random).fit(document.train.word2vec_for_kmeans)

    word2vec_centroids = kmeans_model.cluster_centers_

    document.train.word2vec_clusters = kmeans_model.labels_
    document.validation.word2vec_clusters = kmeans_model.predict(document.validation.word2vec_for_kmeans)
    document.test.word2vec_clusters = kmeans_model.predict(document.test.word2vec_for_kmeans)

    if print_sentences:
        print(f'Clusters number = {args.k}')
        word2vec_clusters_dict = document.train.clusters_to_sentences_indexes_dict(document.train.word2vec_clusters,
                                                                                   args.k)
        print_sentences_by_clusters(document, word2vec_clusters_dict,document.test.word2vec_clusters)
    if t_sne:
        make_tsne(document.train.word2vec_for_kmeans, 'Word2Vec',document.train.word2vec_clusters)

    return word2vec_centroids


def tfidf_kmeans(document,args,tfidf_trained_model, print_sentences=False, t_sne=False):

    document.train.make_tfidf(tfidf_trained_model)
    document.validation.make_tfidf(tfidf_trained_model)
    document.test.make_tfidf(tfidf_trained_model)

    kmeans_tfidf_model = KMeans(n_clusters=args.k,
                                random_state=args.random).fit(document.train.tfidf)

    document.train.tfidf_clusters = kmeans_tfidf_model.labels_
    document.validation.tfidf_clusters = kmeans_tfidf_model.predict(document.validation.tfidf)
    document.test.tfidf_clusters = kmeans_tfidf_model.predict(document.test.tfidf)

    tfidf_clusters_dict = document.train.clusters_to_sentences_indexes_dict(document.train.tfidf_clusters,args.k)

    if print_sentences:
        print(f'Clusters number = {args.k}\n')
        print_sentences_by_clusters(document, tfidf_clusters_dict,
                                    document.test.tfidf_clusters)
        if t_sne:
            make_tsne(document.train.tfidf, 'TF-IDF', document.train.tfidf_clusters)

    return kmeans_tfidf_model.cluster_centers_