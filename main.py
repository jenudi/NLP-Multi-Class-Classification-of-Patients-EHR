# %% Imports

from classes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)}, style="darkgrid")



# %% Initialization
class NLPargs:

    def __init__(self, k=30, min=0.0, random=0, vec_size=300):
        self.k = k
        self.min = min
        self.random = random
        self.windows = [3, 5]
        self.hyperp_lambdas = [0, 0.5, 1]
        self.vec_size = vec_size
        self.doc = None


args = NLPargs()
args.doc = init_classes()


def print_sentences_by_clusters(args, clusters_dict, validation_predict):
    for key in clusters_dict.keys():
        if key in validation_predict:
            validation_sentences_indexes_in_cluster = [index for index, value in enumerate(validation_predict) if
                                                       value == key]
            print(
                f'Validation sentences in cluster number {key + 1}, number of sentences: {len(validation_sentences_indexes_in_cluster)}\n')
            for sentence_index in validation_sentences_indexes_in_cluster:
                print(args.doc.test.get_original_sentences()[sentence_index])
        else:
            print(f'No validation sentences in cluster number {key + 1}\n')
        print('\n')
        print(f'Train sentences in cluster number {key + 1}, size of cluster: {len(clusters_dict[key])} \n')
        sentences_printed = 0
        for index in clusters_dict[key]:
            print(args.doc.train.get_original_sentences()[index])
            sentences_printed += 1
            if sentences_printed > 15:
                break
        print('\n')
        print('\n')


def make_tsne(model, model_name, labels, centers=None, w=None, h=None):
    fig = plt.figure()
    tsne = TSNE()
    palette = sns.color_palette("icefire", len(set(labels)))
    model_tsne_embedded = tsne.fit_transform(model)
    sns.scatterplot(x=model_tsne_embedded[:, 0], y=model_tsne_embedded[:, 1], legend='full', palette=palette, hue=labels)
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    if (w is not None) and (h is not None):
        fig.suptitle(f'Model: {model_name}, Window size: {w}, lambda {h}', fontsize=16)
    else:
        fig.suptitle(f'Model: {model_name}', fontsize=16)
    plt.show()


# %% TF-IDF
def run_tfidf_model(args, t_sne=False):
    tfidf_model = TfidfVectorizer(min_df=args.min,
                                  smooth_idf=True,
                                  norm='l1')

    tfidf_trained = tfidf_model.fit(args.doc.train.get_sentences())

    args.doc.train.make_tfidf(tfidf_trained)

    # args.doc.validation.make_tfidf(tfidf_trained)
    args.doc.test.make_tfidf(tfidf_trained)

    kmeans_tfidf_model = KMeans(n_clusters=args.k,
                                random_state=args.random).fit(args.doc.train.tfidf)

    args.doc.train.tfidf_clusters_labels = kmeans_tfidf_model.labels_
    # args.doc.validation.tfidf_clusters_labels = kmeans_tfidf_model.predict(args.doc.validation.tfidf)
    args.doc.test.tfidf_clusters_labels = kmeans_tfidf_model.predict(args.doc.test.tfidf)

    tfidf_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(args.doc.train.tfidf_clusters_labels,
                                                                            args.k)
    if __name__ == "__main__":
        if t_sne:
            make_tsne(args.doc.train.tfidf, 'TF-IDF', args.doc.train.tfidf_clusters_labels,
                      centers=kmeans_tfidf_model.cluster_centers_, w=None, h=None)
        else:
            print(f'Clusters number = {args.k}\n')
            print_sentences_by_clusters(args, tfidf_clusters_dict,
                                        args.doc.test.tfidf_clusters_labels)

    return kmeans_tfidf_model.cluster_centers_


# %% Word2Vec
def run_word2vec_model(args, t_sne=False):
    train_tokens = args.doc.train.get_sentences_tokens()
    # train_tokens.append(['un-known'])
    word2vec_centroids = dict()

    for window in args.windows:
        word2vec_model = Word2Vec(min_count=args.min,
                                  window=window,
                                  size=args.vec_size,
                                  sample=1e-3,
                                  alpha=0.03,
                                  min_alpha=0.0007)

        word2vec_model.build_vocab(train_tokens)
        word2vec_model.train(train_tokens,
                             total_examples=word2vec_model.corpus_count,
                             epochs=30)

        for hyperp_lambda in args.hyperp_lambdas:

            args.doc.train.make_word2vec(word2vec_model, hyperp_lambda, window)
            # args.doc.validation.make_word2vec(word2vec_model, hyperp_lambda, window)
            args.doc.test.make_word2vec(word2vec_model, hyperp_lambda, window)

            kmeans_word2vec_model = KMeans(n_clusters=args.k,
                                           random_state=args.random).fit(
                args.doc.train.word2vec[(hyperp_lambda,window)])

            word2vec_centroids[(hyperp_lambda, window)] = kmeans_word2vec_model.cluster_centers_

            args.doc.train.word2vec_clusters_labels[(hyperp_lambda, window)] = kmeans_word2vec_model.labels_
            # args.doc.validation.word2vec_clusters_labels[(hyperp_lambda, window)] = kmeans_word2vec_model.predict(
            #   args.doc.validation.word2vec[(hyperp_lambda, window)])
            args.doc.test.word2vec_clusters_labels[(hyperp_lambda, window)] = kmeans_word2vec_model.predict(
                args.doc.test.word2vec[(hyperp_lambda, window)])

            word2vec_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(
                args.doc.train.word2vec_clusters_labels[(hyperp_lambda, window)], args.k)

            if __name__ == "__main__":
                if t_sne:
                    make_tsne(args.doc.train.word2vec[(hyperp_lambda, window)], 'Word2Vec', \
                              args.doc.train.word2vec_clusters_labels[(hyperp_lambda, window)], centers=None, w=window,
                              h=hyperp_lambda)
                else:
                    print(f'Clusters number = {args.k}, lambda = {hyperp_lambda}, window size = {window} \n')
                    # print_sentences_by_clusters(args, word2vec_clusters_dict,
                    #                           args.doc.validation.word2vec_clusters_labels[(hyperp_lambda, window)])
                    print_sentences_by_clusters(args, word2vec_clusters_dict,
                                                args.doc.test.word2vec_clusters_labels[(hyperp_lambda, window)])

    return word2vec_centroids


# %%

def run_word2vec_pubmed_model(args,t_sne=False):
    try:
        word2vec_pubmed_model = KeyedVectors.load_word2vec_format('PubMed-w2v.bin', binary=True)
        # word2vec_pubmed_model = Word2Vec.load("word2vec_pubmed.model")
    except FileNotFoundError:
        print('No such model file')
        return

    word2vec_pubmed_centroids = dict()

    for hyperp_lambda in args.hyperp_lambdas:

        args.doc.train.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)
        args.doc.validation.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)
        # args.doc.test.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)

        kmeans_word2vec_pubmed_model = KMeans(n_clusters=args.k, random_state=args.random).fit(
            args.doc.train.word2vec_pubmed[hyperp_lambda])

        word2vec_pubmed_centroids[hyperp_lambda] = kmeans_word2vec_pubmed_model.cluster_centers_

        args.doc.train.word2vec_pubmed_clusters_labels[hyperp_lambda] = kmeans_word2vec_pubmed_model.labels_
        args.doc.validation.word2vec_pubmed_clusters_labels[hyperp_lambda] = kmeans_word2vec_pubmed_model.predict(
            args.doc.validation.word2vec_pubmed[hyperp_lambda])
        # args.doc.test.word2vec_pubmed_clusters_labels[hyperp_lambda] = kmeans_word2vec_pubmed_model.predict(
        #   args.doc.test.word2vec_pubmed[hyperp_lambda])

        word2vec_pubmed_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(
            args.doc.train.word2vec_pubmed_clusters_labels[hyperp_lambda], args.k)

        if __name__ == "__main__":
            if t_sne:
                make_tsne(args.doc.train.word2vec_pubmed[hyperp_lambda], 'Word2Vec Pubmed', \
                          args.doc.train.word2vec_pubmed_clusters_labels[hyperp_lambda], centers=None, w=None,
                          h=hyperp_lambda)
            else:
                print(f'Clusters number = {args.k}, lambda = {hyperp_lambda} \n')
                print_sentences_by_clusters(args, word2vec_pubmed_clusters_dict,
                                            args.doc.validation.word2vec_pubmed_clusters_labels[hyperp_lambda])
    return word2vec_pubmed_centroids


# %%

tfidf_centroids = run_tfidf_model(args,t_sne=True)


word2vec_centroids=run_word2vec_model(args,t_sne=True)
word2vec_pubmed_centroids=run_word2vec_pubmed_model(args,t_sne=True)





