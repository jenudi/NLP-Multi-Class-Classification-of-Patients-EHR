# %% Imports

from classes import *
from RNN import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch.optim as optim

from rnn_utils import make_embbedings
from rnn_utils import make_random_sample

sns.set(rc={'figure.figsize': (11.7, 8.27)}, style="darkgrid")


# %% Initialization
def init_classes(threshold=0):
    data = make_data(threshold)
    return make_preprocess(data)


def make_data(threshold_for_dropping=0):
    encounter_data = pd.read_csv("encounter.csv").rename(str.lower, axis='columns')
    data = encounter_data.loc[:, ['soap_note', 'cc']]
    data.dropna(inplace=True, subset=['soap_note'])
    data.reset_index(drop=True, inplace=True)
    data['cc'].fillna('no specific issue', inplace=True)
    data['cc'] = data['cc'].str.lower()
    if threshold_for_dropping > 0:
        temp_dict = data['cc'].value_counts().to_dict()
        temp_list = [index for index, rare_labels in enumerate(data['cc'].values)
                     if temp_dict[rare_labels] <= threshold_for_dropping]
        data.drop(temp_list, inplace=True)
        data.reset_index(drop=True, inplace=True)

    data.sort_values('cc',inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def make_preprocess(data):
    soap = data['soap_note']
    soap_temp = [re.split('o:''|''o :', i) for i in soap]  # split by "o:" or "o :"
    temp_sentences = [i[0].strip().strip('s:').lower() for i in soap_temp]
    try:
        _ = stopwords.words("english")
    except LookupError:
        import nltk
        nltk.download('stopwords')
    stopword_set = set(stopwords.words("english"))
    document = Document('\n '.join(temp_sentences))
    document.do_replaces()
    document.make_sentences('\n ')
    for index,sentence in enumerate(document.sentences):
        sentence.label=data['cc'][index]
        sentence.stem_and_check_stop(stopword_set)
        sentence.make_tokens()
        sentence.make_original_text_tokens()
        sentence.text = ' '.join(sentence.tokens)
    document.train_test_split()
    document.train.make_lexicon()
    print('Classes are ready to use.')
    return document


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


def make_tsne(model, model_name, labels, w=None, h=None):
    fig = plt.figure()
    tsne = TSNE()
    palette = sns.color_palette("icefire", len(set(labels)))
    model_tsne_embedded = tsne.fit_transform(model)
    temp_df = pd.DataFrame({'x1':model_tsne_embedded[:, 0],'x2':model_tsne_embedded[:, 1],'y':labels})
    centers = temp_df.groupby(by=["y"]).mean()
    sns.scatterplot(x=model_tsne_embedded[:, 0], y=model_tsne_embedded[:, 1], legend='full', palette=palette, hue=labels)
    plt.scatter(centers.iloc[:,0].values, centers.iloc[:,1].values, c=palette, s=200, alpha=0.5)
    if (w is not None) and (h is not None):
        fig.suptitle(f'Model: {model_name}, Window size: {w}, lambda {h}', fontsize=16)
    elif h is not None:
        fig.suptitle(f'Model: {model_name}, lambda {h}', fontsize=16)
    else:
        fig.suptitle(f'Model: {model_name}', fontsize=16)
    plt.show()


#%% RNN
def train(input_tensor, cls_numbers):
    hidden = rnn.init_hidden()
    output = None
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i], hidden)
    loss = criterion(output, cls_numbers)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


def init_rnn(n_iters=100000):
    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    for i in range(n_iters):
        label, sentence,input_tensor, cls_numbers = make_random_sample(args,embbedings_model)
        cls_numbers = torch.tensor(cls_numbers)
        cls_n = torch.reshape(cls_numbers, (-1,))
        output, loss = train(input_tensor, cls_n)
        current_loss += loss
        print(f"index: {i}, loss: {loss}")
        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0
        if (i + 1) % print_steps == 0:
            guess = args.doc.train.labels_dict[int(torch.max(output, 1)[1].detach())]
            correct = "CORRECT" if guess == label else f"WRONG ({label})"
            print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {sentence} / {guess} {correct}")

    plt.figure()
    plt.plot(all_losses)
    plt.show()



# %% TF-IDF
def tfidf_kmeans(args, t_sne=False):
    tfidf_model = TfidfVectorizer(min_df=args.min,
                                  smooth_idf=True,
                                  norm='l1')

    tfidf_trained = tfidf_model.fit(args.doc.train.get_sentences())

    args.doc.train.make_tfidf(tfidf_trained)
    args.doc.validation.make_tfidf(tfidf_trained)
    args.doc.test.make_tfidf(tfidf_trained)

    kmeans_tfidf_model = KMeans(n_clusters=args.k,
                                random_state=args.random).fit(args.doc.train.tfidf)

    args.doc.train.tfidf_clusters = kmeans_tfidf_model.labels_
    args.doc.validation.tfidf_clusters = kmeans_tfidf_model.predict(args.doc.validation.tfidf)
    args.doc.test.tfidf_clusters = kmeans_tfidf_model.predict(args.doc.test.tfidf)

    tfidf_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(args.doc.train.tfidf_clusters,args.k)

    if __name__ == "__main__":
        print(f'Clusters number = {args.k}\n')
        print_sentences_by_clusters(args, tfidf_clusters_dict,
                                    args.doc.test.tfidf_clusters)
        if t_sne:
            make_tsne(args.doc.train.tfidf, 'TF-IDF', args.doc.train.tfidf_clusters, w=None, h=None)

    return kmeans_tfidf_model.cluster_centers_


# %% Word2Vec
def word2vec_kmeans(args, t_sne=False):
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
            args.doc.validation.make_word2vec(word2vec_model, hyperp_lambda, window)
            args.doc.test.make_word2vec(word2vec_model, hyperp_lambda, window)

            kmeans_word2vec_model = KMeans(n_clusters=args.k,
                                           random_state=args.random).fit(
                args.doc.train.word2vec[(hyperp_lambda,window)])

            word2vec_centroids[(hyperp_lambda, window)] = kmeans_word2vec_model.cluster_centers_

            args.doc.train.word2vec_clusters[(hyperp_lambda, window)] = kmeans_word2vec_model.labels_
            args.doc.validation.word2vec_clusters[(hyperp_lambda, window)] = kmeans_word2vec_model.predict(
                args.doc.validation.word2vec[(hyperp_lambda, window)])
            args.doc.test.word2vec_clusters[(hyperp_lambda, window)] = kmeans_word2vec_model.predict(
                args.doc.test.word2vec[(hyperp_lambda, window)])

            word2vec_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(
                args.doc.train.word2vec_clusters[(hyperp_lambda, window)], args.k)

            if __name__ == "__main__":
                print(f'Clusters number = {args.k}, lambda = {hyperp_lambda}, window size = {window} \n')
                print_sentences_by_clusters(args, word2vec_clusters_dict,
                                            args.doc.test.word2vec_clusters[(hyperp_lambda, window)])
                if t_sne:
                    make_tsne(args.doc.train.word2vec[(hyperp_lambda, window)], 'Word2Vec', \
                              args.doc.train.word2vec_clusters[(hyperp_lambda, window)],w=window, h=hyperp_lambda)


    return word2vec_centroids


def word2vec_pubmed_kmeans(args,t_sne=False):
    try:
        word2vec_pubmed_model = KeyedVectors.load_word2vec_format('PubMed-w2v.bin', binary=True)
        # word2vec_pubmed_model = Word2Vec.load("word2vec_pubmed.model")
    except FileNotFoundError:
        print('No such model file')
        return

    word2vec_pubmed_centroids = dict()

    for hyperp_lambda in args.hyperp_lambdas:

        args.doc.train.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)
        args.doc.test.make_word2vec_pubmed(word2vec_pubmed_model, hyperp_lambda)

        kmeans_word2vec_pubmed_model = KMeans(n_clusters=args.k, random_state=args.random).fit(
            args.doc.train.word2vec_pubmed[hyperp_lambda])

        word2vec_pubmed_centroids[hyperp_lambda] = kmeans_word2vec_pubmed_model.cluster_centers_

        args.doc.train.word2vec_pubmed_clusters[hyperp_lambda] = kmeans_word2vec_pubmed_model.labels_
        args.doc.test.word2vec_pubmed_clusters[hyperp_lambda] = kmeans_word2vec_pubmed_model.predict(args.doc.test.word2vec_pubmed[hyperp_lambda])

        word2vec_pubmed_clusters_dict = args.doc.train.clusters_to_sentences_indexes_dict(
            args.doc.train.word2vec_pubmed_clusters[hyperp_lambda], args.k)

        if __name__ == "__main__":
            print(f'Clusters number = {args.k}, lambda = {hyperp_lambda} \n')
            print_sentences_by_clusters(args, word2vec_pubmed_clusters_dict,
                                        args.doc.test.word2vec_pubmed_clusters[hyperp_lambda])
            if t_sne:
                make_tsne(args.doc.train.word2vec_pubmed[hyperp_lambda], 'Word2Vec Pubmed', \
                          args.doc.train.word2vec_pubmed_clusters[hyperp_lambda], w=None, h=hyperp_lambda)

    return word2vec_pubmed_centroids


#%%

args = NLPargs(k=30, min=0.0, random=0, vec_size=300, hidden=350,min_cls=5, lr=0.0005)
args.doc = init_classes(args.min_cls)

args.doc.train.make_labels_dict_and_weights()
embbedings_model = make_embbedings(args)
rnn = RNN(args.vec_size, args.hidden, len(args.doc.train.labels_dict))
criterion = nn.NLLLoss(weight=args.doc.train.weights)
optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr)
init_rnn(n_iters=100000)

#%%
tfidf_centroids = tfidf_kmeans(args,t_sne=True)
word2vec_centroids=word2vec_kmeans(args,t_sne=True)
word2vec_pubmed_centroids=word2vec_pubmed_kmeans(args,t_sne=True)

# %% change according to the final chosen word2vec model
word2vec_chosen_params=(0,5)