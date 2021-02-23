# %% Imports

from classes import *
from RNN import *
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
    print('Classes are ready to use\n')
    return document


#%% initializing NLP arguments and data
args = NLP_args(k=30, min=0.0, random=0, vec_size=300, hidden=350,min_cls=5, lr=0.0005)
data = make_data(threshold_for_dropping=0)
document=preprocess_data(data)


#%% word2vec kmeans

word2vec_window_5_model = Word2Vec(min_count=args.min,
                                   window=5,
                                   size=args.vec_size,
                                   sample=1e-3,
                                   alpha=0.03,
                                   min_alpha=0.0007)


word2vec_chosen_model=word2vec_window_5_model
word2vec_chosen_vector_size= args.vec_size

filename = "word2vec_model.sav"
pickle.dump(word2vec_chosen_model, open(filename, "wb"))

train_tokens = document.train.get_sentences_tokens()
word2vec_chosen_model.build_vocab(train_tokens)
word2vec_chosen_model.train(train_tokens, total_examples=word2vec_chosen_model.corpus_count, epochs=30)
word2vec_centroids=word2vec_kmeans(document,args,word2vec_chosen_model, word2vec_chosen_vector_size)


# %% RNN classification

eval_best_rnn_model(args,document)

'''
document.train.make_word2vec_for_rnn(word2vec_chosen_model)
document.validation.make_word2vec_for_rnn(word2vec_chosen_model)
document.test.make_word2vec_for_rnn(word2vec_chosen_model)


rnn_model = RNN(args.vec_size, args.hidden, len(document.train.labels_dict))
criterion = nn.NLLLoss(weight=document.train.weights)
optimizer = torch.optim.SGD(rnn_model.parameters(), lr=args.lr)

init_rnn(rnn_model,criterion,optimizer,document,n_iters=100000)

torch.save(rnn_model.state_dict(),'rnn_model.pth')
rnn_model.load_state_dict(torch.load('rnn_model.pth'))
rnn_model.eval()
predict_rnn(document,rnn_model)
'''

#%%TF-IDF kmeans
random.shuffle(document.train.sentences)
random.shuffle(document.test.sentences)
random.shuffle(document.validation.sentences)


tfidf_model = TfidfVectorizer(min_df=args.min,smooth_idf=True,norm='l1')
tfidf_trained = tfidf_model.fit(document.train.get_sentences())
tfidf_centroids = tfidf_kmeans(document,args,tfidf_model)

filename = "tfidf_model.sav"
pickle.dump(tfidf_model, open(filename, "wb"))


#%% random forest classification
train_labels = [sentence.label for sentence in document.train.sentences]
validation_labels = [sentence.label for sentence in document.validation.sentences]
n_estimators_list=[10,100,200,300,400,500,1000]
validation_scores=list()
for n_estimators in n_estimators_list:
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators,criterion='gini',max_depth=None,bootstrap=True,random_state=0)
    _ = random_forest_model.fit(document.train.tfidf,train_labels)
    validation_score=random_forest_model.score(document.validation.tfidf,validation_labels)
    validation_scores.append(validation_score)
    print("Random forest with TF-IDF enbeddings score of n_estimator=" + str(n_estimators)+ " is " + str(validation_score))

best_n_estimator=n_estimators_list[np.argmax(validation_scores)]
random_forest_chosen_model=RandomForestClassifier(n_estimators=n_estimators,criterion='gini',max_depth=None,bootstrap=True,random_state=0)
_ = random_forest_chosen_model.fit(document.train.tfidf, train_labels)

filename = "random_forest_model.sav"
pickle.dump(random_forest_chosen_model, open(filename, "wb"))