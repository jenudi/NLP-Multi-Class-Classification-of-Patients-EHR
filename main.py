from preprocessing import *
from classes import *
from kmeans import *
from RNN import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class InitModels:
    def __init__(self, args, data, document):
        self.args = args
        self.data = data
        self.document = document
        self.tfidf_trained = None

    def init_kmeans(self,model_name="self-trained word2vec with window 5",window=5,save=False,t_sne_figure=True,return_centroids=False):
        word2vec_for_kmeans_model = Word2Vec(min_count=self.args.min,
                                             window=window,
                                             size=self.args.word2vec_vec_size_for_kmeans,
                                             sample=1e-3, alpha=0.03, min_alpha=0.0007)

        train_tokens = self.document.train.get_sentences_tokens()
        _ = word2vec_for_kmeans_model.build_vocab(train_tokens)
        _ = word2vec_for_kmeans_model.train(train_tokens,
                                            total_examples=word2vec_for_kmeans_model.corpus_count,
                                            epochs=30)
        if save:
            word2vec_for_kmeans_model.save("word2vec_for_kmeans_model.model")

        if return_centroids:
            return word2vec_kmeans(self.document, self.args, word2vec_for_kmeans_model,
                                    self.args.word2vec_vec_size_for_kmeans, model_name, t_sne_figure)

        print('k-means initialized')

    def train_rnn(self,continue_training=False, decay_learning=False,save_results=False,print_f1=False,check_test_set=False):
        eval_rnn = pd.DataFrame()
        eval_rnn['y_true'] = [
            list(self.document.labels_dict.keys())[list(self.document.labels_dict.values()).index(sentence.label)]
            if sentence.label in self.document.labels_dict.values()
            else len(self.document.labels_dict.keys()) for sentence in self.document.validation.sentences]

        for model in self.args.models:
            if model == 'w2v_3':
                document.train.make_word2vec_for_rnn(args, 3)
            elif model == 'w2v_5':
                document.train.make_word2vec_for_rnn(args, 5)
            elif model == 'w2v_p':
                args.word2vec_vec_size_for_rnn = 200
                document.train.make_word2vec_for_rnn(args, None)
            rnn = RNN(args.word2vec_vec_size_for_rnn, 42, args.hidden_layer, 2, len(document.labels_dict))
            #rnn = RNN(self.args.word2vec_vec_size_for_rnn, self.args.hidden_layer, len(self.document.labels_dict))
            training = TrainValidate(self.args, self.document, rnn)
            eval_rnn[f'y_pred_{self.document.train.word2vec_model_name}_{self.args.lr}_{self.args.hidden_layer}'] = training.main(
                continue_training, decay_learning)

        if save_results:
            eval_rnn.to_csv('rnn_results.csv',index=False)
        if print_f1:
            for i in range(1, 4):
                print(f"Model: {eval_rnn.columns[i]}. micro f1: {f1_score(eval_rnn.iloc[:, 0], eval_rnn.iloc[:, i], average='micro')}")
        test_data_set = RecordsDataset(dataset=self.document.test, doc=self.document)
        test_dl = DataLoader(test_data_set, batch_size=10, shuffle=False)
        if check_test_set:
            test_pred = []
            for batch_ndx, batch_tup in enumerate(test_dl, 0):
                with torch.no_grad():
                    rnn.eval()
                    input, _ = batch_tup
                    output = rnn(input.squeeze(2))
                    output = torch.max(F.softmax(output.detach(), dim=1), 1)[1]
                    for i in output:
                        test_pred.append(int(i))
            test_true = [
                list(self.document.labels_dict.keys())[list(self.document.labels_dict.values()).index(sentence.label)]
                if sentence.label in self.document.labels_dict.values()
                else len(self.document.labels_dict.keys()) for sentence in self.document.test.sentences]
            print(f1_score(test_pred, test_true, average='micro'))

    def init_tfidf(self,t_sne_figure=True,return_centroids=False,save=False):
        random.shuffle(self.document.train.sentences)
        random.shuffle(self.document.test.sentences)
        random.shuffle(self.document.validation.sentences)

        tfidf_model = TfidfVectorizer(min_df=self.args.min, smooth_idf=True, norm='l1')
        self.tfidf_trained = tfidf_model.fit(self.document.train.get_sentences())
        if save:
            pickle.dump(tfidf_model, open("tfidf_model.pkl", "wb"))
        if return_centroids:
            return tfidf_kmeans(self.document, self.args, tfidf_model, t_sne_figure)
        print('tf-idf initialized')

    def train_random_forest(self):
        if self.tfidf_trained is None:
            print('no tfidf model')
            return
        self.document.validation.make_tfidf(self.tfidf_trained)
        train_labels = [sentence.label for sentence in self.document.train.sentences]
        validation_labels = [sentence.label for sentence in self.document.validation.sentences]
        n_estimators_list = [100, 200, 300, 400, 500, 1000, 3000, 5000]
        random_forest_validation_scores = list()

        for n_estimators in n_estimators_list:
            random_forest_model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None,
                                                         bootstrap=True, class_weight="balanced_subsample",
                                                         random_state=0)
            _ = random_forest_model.fit(self.document.train.tfidf, train_labels)
            predicted_validation_labels = random_forest_model.predict(self.document.validation.tfidf)
            random_forest_score = f1_score(validation_labels, predicted_validation_labels, average='micro')
            random_forest_validation_scores.append(random_forest_score)
            print("Random forest with TF-IDF embeddings score of n_estimator=" + str(n_estimators) + " is " + str(
                random_forest_score))

        best_n_estimator = n_estimators_list[np.argmax(random_forest_validation_scores)]
        print("the best n_estimator is " + str(best_n_estimator) + " with score of " + str(
            max(random_forest_validation_scores)))
        chosen_random_forest_model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None,
                                                            bootstrap=True, class_weight="balanced_subsample",
                                                            random_state=0)
        _ = chosen_random_forest_model.fit(self.document.train.tfidf, train_labels)

        self.document.test.make_tfidf(self.tfidf_trained)
        test_labels = [sentence.label for sentence in self.document.test.sentences]
        predicted_test_labels = chosen_random_forest_model.predict(self.document.test.tfidf)
        random_forest_test_score = f1_score(test_labels, predicted_test_labels, average='micro')
        print("test score of random forest is " + str(random_forest_test_score))

        pickle.dump(chosen_random_forest_model, open("random_forest_model.pkl", "wb"))


#%% initializing NLP arguments and data
if __name__ == '__main__':
    args = NLP_args(k=30, min=0.0, random=0,min_cls=5,lr=0.001, hidden_layer=300,epoch_num=500, l2=0.09)
    data = make_data(threshold_for_dropping=args.min_cls)
    document=preprocess_data(data)
    models = InitModels(args,data,document)
    document.train.make_word2vec_for_rnn(args, 5)
    rnn = RNN(args.word2vec_vec_size_for_rnn, 42, args.hidden_layer, 2, len(document.labels_dict))
    training = TrainValidate(args, document, rnn)
    #training.main()
    word2vec_centroids = models.init_kmeans(model_name="self-trained word2vec with window 5",
                                            window=5, save=True, t_sne_figure=False,
                                            return_centroids=True)
    tfidf_centroids = models.init_tfidf(t_sne_figure=False,return_centroids=True,save=True)

