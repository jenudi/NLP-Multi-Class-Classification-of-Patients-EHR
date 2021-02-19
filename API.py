from flask import Flask, request, jsonify
from classes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors


app = Flask(__name__)


@app.route('/tfidf')
def tfidf_representation(sentence):
    pass


@app.route('/tfidf')
def word2vec_representation(sentence):
    pass


try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')
stopword_set = set(stopwords.words("english"))


def sentence_preprocess(sentence,stopword_set):
    split_sentence = re.split('o:''|''o :', sentence)
    preprocessed_sentence = Sentence_in_document(split_sentence[0].strip().strip('s:').lower())
    preprocessed_sentence.do_replaces()
    preprocessed_sentence.stem_and_check_stop(stopword_set)
    preprocessed_sentence.make_tokens()
    preprocessed_sentence.make_original_text_tokens()
    preprocessed_sentence.text = ' '.join(sentence.tokens)


if __name__ == "__main__":
    app.run()