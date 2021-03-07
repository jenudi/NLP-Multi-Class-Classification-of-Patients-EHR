# %% Imports

from RNN import *
from classes import *
from kmeans import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import random
import pickle
from sklearn.metrics import f1_score

sns.set(rc={'figure.figsize': (11.7, 8.27)}, style="darkgrid")


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


def preprocess_data(data):
    soap = data['soap_note']
    soap_temp = [re.split('o:''|''o :', i) for i in soap]  # split by "o:" or "o :"
    temp_sentences = [i[0].strip().strip('s:').lower() for i in soap_temp]
    try:
        _ = stopwords.words("english")
    except LookupError:
        import nltk
        nltk.download('stopwords')
    stopword_set = set(stopwords.words("english"))
    document_ = Document('\n '.join(temp_sentences))
    document_.do_replaces()
    document_.make_sentences('\n ')
    for index,sentence in enumerate(document_.sentences):
        sentence.label=data['cc'][index]
        sentence.stem_and_check_stop(stopword_set)
        sentence.make_tokens()
        sentence.make_original_text_tokens()
        sentence.text = ' '.join(sentence.tokens)
    document_.train_test_split()
    document_.train.make_lexicon()
    document_.make_dict()
    pickle.dump(document_.labels_dict,open("labels_dict.pkl", "wb"))
    print('Classes are ready to use\n')
    return document_


#%% initializing NLP arguments and data

args = NLP_args(k=30, min=0.0, random=0,min_cls=5,lr=0.001, hidden_layer=400,epoch_num=30)
data = make_data(threshold_for_dropping=args.min_cls)
document=preprocess_data(data)


#%% word2vec kmeans
word2vec_for_kmeans_model = Word2Vec(min_count=args.min,
                                    window=5,
                                    size=args.word2vec_vec_size_for_kmeans,
                                    sample=1e-3,
                                    alpha=0.03,
                                    min_alpha=0.0007)


train_tokens = document.train.get_sentences_tokens()
_=word2vec_for_kmeans_model.build_vocab(train_tokens)
_=word2vec_for_kmeans_model.train(train_tokens, total_examples=word2vec_for_kmeans_model.corpus_count, epochs=30)

word2vec_centroids=word2vec_kmeans(document,args,word2vec_for_kmeans_model, args.word2vec_vec_size_for_kmeans)

word2vec_for_kmeans_model.save("word2vec_for_kmeans_model.model")



# %% RNN classification

eval_rnn = pd.DataFrame()
eval_rnn['y_true'] = [list(document.labels_dict.keys())[list(document.labels_dict.values()).index(sentence.label)]
                          if sentence.label in document.labels_dict.values()
                          else len(document.labels_dict.keys()) for sentence in document.validation.sentences]

document.train.make_word2vec_for_rnn(args, 5)
rnn = RNN(args.word2vec_vec_size_for_rnn, args.hidden_layer, len(document.labels_dict))
#rnn = RNN(args.word2vec_vec_size_for_rnn,args.hidden_layer, 2, len(document.labels_dict))
training = TrainValidate(args,document,rnn)
eval_rnn[f'y_pred_{document.train.word2vec_model_name}_{args.lr}_{args.hidden_layer}'] = training.main(continue_training=False, decay_learning=False)



for model in args.models:
    if model == 'w2v_3':
        document.train.make_word2vec_for_rnn(args, 3)
    elif model == 'w2v_5':
        document.train.make_word2vec_for_rnn(args, 5)
    elif model == 'w2v_p':
        args.word2vec_vec_size_for_rnn = 200
        document.train.make_word2vec_for_rnn(args, None)


document.train.make_word2vec_for_rnn(args, 5)
rnn = RNN(args.word2vec_vec_size_for_rnn, args.hidden_layer, len(document.labels_dict))
#rnn = RNN(args.word2vec_vec_size_for_rnn,args.hidden_layer, 2, len(document.labels_dict))
training = TrainValidate(args,document,rnn)
eval_rnn[f'y_pred_{document.train.word2vec_model_name}_{args.lr}_{args.hidden_layer}'] = training.main(continue_training=False, decay_learning=False)


document.train.make_word2vec_for_rnn(args,5)
document.train.word2vec_for_rnn.save("word2vec_for_rnn_model.model")



#%%TF-IDF kmeans
random.shuffle(document.train.sentences)
random.shuffle(document.test.sentences)
random.shuffle(document.validation.sentences)


tfidf_model = TfidfVectorizer(min_df=args.min,smooth_idf=True,norm='l1')
tfidf_trained = tfidf_model.fit(document.train.get_sentences())
tfidf_centroids = tfidf_kmeans(document,args,tfidf_model)

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
    #validation_score=random_forest_model.score(document.validation.tfidf,validation_labels)
    y_pred=random_forest_model.predict(document.validation.tfidf)
    random_forest_score=f1_score(validation_labels,y_pred,average='micro')
    random_forest_validation_scores.append(random_forest_score)
    print("Random forest with TF-IDF enbeddings score of n_estimator=" + str(n_estimators)+ " is " + str(random_forest_score))

best_n_estimator=n_estimators_list[np.argmax(random_forest_validation_scores)]
print("the best n_estimator is "+ str(best_n_estimator) + " with score of "+ str(max(random_forest_validation_scores)))
chosen_random_forest_model=RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, bootstrap=True, class_weight="balanced_subsample", random_state=0)
_ = chosen_random_forest_model.fit(document.train.tfidf, train_labels)

pickle.dump(chosen_random_forest_model, open("random_forest_model.pkl", "wb"))

document.test.make_tfidf(tfidf_trained)
test_labels = [sentence.label for sentence in document.test.sentences]
y_pred = chosen_random_forest_model.predict(document.test.tfidf)
random_forest_test_score=f1_score(test_labels,y_pred,average='micro')
print("test score of random forest is " + str(random_forest_test_score))