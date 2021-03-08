from preprocessing import *
from classes import *
from kmeans import *
from RNN import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score



#%% initializing NLP arguments and data
args = NLP_args(k=30, min=0.0, random=0,min_cls=5,lr=0.001, hidden_layer=400,epoch_num=50)
data = make_data(threshold_for_dropping=args.min_cls)
document=preprocess_data(data)


#%% word2vec kmeans
word2vec_for_kmeans_model = Word2Vec(min_count=args.min,window=5,size=args.word2vec_vec_size_for_kmeans,
                                    sample=1e-3,alpha=0.03,min_alpha=0.0007)

train_tokens = document.train.get_sentences_tokens()
_=word2vec_for_kmeans_model.build_vocab(train_tokens)
_=word2vec_for_kmeans_model.train(train_tokens, total_examples=word2vec_for_kmeans_model.corpus_count, epochs=30)

model_name="self-trained word2vec with window 5"
word2vec_centroids=word2vec_kmeans(document,args,word2vec_for_kmeans_model, args.word2vec_vec_size_for_kmeans,model_name,t_sne=True)

word2vec_for_kmeans_model.save("word2vec_for_kmeans_model.model")


# %% RNN classification
eval_rnn = pd.DataFrame()
eval_rnn['y_true'] = [list(document.labels_dict.keys())[list(document.labels_dict.values()).index(sentence.label)]
                          if sentence.label in document.labels_dict.values()
                          else len(document.labels_dict.keys()) for sentence in document.validation.sentences]


for model in args.models:
    if model == 'w2v_3':
        document.train.make_word2vec_for_rnn(args, 3)
    elif model == 'w2v_5':
        document.train.make_word2vec_for_rnn(args, 5)
    elif model == 'w2v_p':
        args.word2vec_vec_size_for_rnn = 200
        document.train.make_word2vec_for_rnn(args, None)
    rnn = RNN(args.word2vec_vec_size_for_rnn, args.hidden_layer, len(document.labels_dict))
    training = TrainValidate(args,document,rnn)
    eval_rnn[f'y_pred_{document.train.word2vec_model_name}_{args.lr}_{args.hidden_layer}'] = training.main(continue_training=False,
                                                                                                           decay_learning=False)

#random_forest_test_score=f1_score(eval_rnn.iloc[:,0], eval_rnn.iloc[:,1], average='micro')


#%%TF-IDF kmeans
random.shuffle(document.train.sentences)
random.shuffle(document.test.sentences)
random.shuffle(document.validation.sentences)


tfidf_model = TfidfVectorizer(min_df=args.min,smooth_idf=True,norm='l1')
tfidf_trained = tfidf_model.fit(document.train.get_sentences())
tfidf_centroids = tfidf_kmeans(document,args,tfidf_model,t_sne=True)

pickle.dump(tfidf_model, open("tfidf_model.pkl", "wb"))


#%% random forest classification
document.validation.make_tfidf(tfidf_trained)
train_labels = [sentence.label for sentence in document.train.sentences]
validation_labels = [sentence.label for sentence in document.validation.sentences]
n_estimators_list=[100,200,300,400,500,1000,3000,5000]
random_forest_validation_scores=list()
for n_estimators in n_estimators_list:
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, bootstrap=True, class_weight="balanced_subsample", random_state=0)
    _ = random_forest_model.fit(document.train.tfidf,train_labels)
    predicted_validation_labels=random_forest_model.predict(document.validation.tfidf)
    random_forest_score=f1_score(validation_labels,predicted_validation_labels,average='micro')
    random_forest_validation_scores.append(random_forest_score)
    print("Random forest with TF-IDF enbeddings score of n_estimator=" + str(n_estimators)+ " is " + str(random_forest_score))

best_n_estimator=n_estimators_list[np.argmax(random_forest_validation_scores)]
print("the best n_estimator is "+ str(best_n_estimator) + " with score of "+ str(max(random_forest_validation_scores)))
chosen_random_forest_model=RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, bootstrap=True, class_weight="balanced_subsample", random_state=0)
_ = chosen_random_forest_model.fit(document.train.tfidf, train_labels)

pickle.dump(chosen_random_forest_model, open("random_forest_model.pkl", "wb"))

document.test.make_tfidf(tfidf_trained)
test_labels = [sentence.label for sentence in document.test.sentences]
predicted_test_labels = chosen_random_forest_model.predict(document.test.tfidf)
random_forest_test_score=f1_score(test_labels,predicted_test_labels,average='micro')
print("test score of random forest is " + str(random_forest_test_score))