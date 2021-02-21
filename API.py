from flask import Flask, request, jsonify
from classes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from gensim.models.keyedvectors import KeyedVectors


app = Flask(__name__)

try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')
stopword_set = set(stopwords.words("english"))

@app.route("/tfidf/<str:sentence>")
def tfidf_representation(sentence):

    if sentence is None:
        return jsonify({"error": "no sentence"})
    elif not isinstance(sentence, str):
        return jsonify({"error": "value entered is not a string"})

    try:
        preprocessed_sentence = sentence_preprocess(sentence, stopword_set)
    except:
        return jsonify({"error: sentence could not be preprocessed"})


@app.route("/word2vec/<str:sentence>")
def word2vec_representation(sentence):

    if sentence is None:
        return jsonify({"error":"no sentence"})
    elif not isinstance(sentence,str):
        return jsonify({"error":"value entered is not a string"})

    try:
        preprocessed_sentence = sentence_preprocess(sentence,stopword_set)
    except:
        return jsonify({"error: sentence could not be preprocessed"})



def sentence_preprocess(sentence,stopword_set):
    split_sentence = re.split('o:''|''o :', sentence)
    preprocessed_sentence = Sentence_in_document(split_sentence[0].strip().strip('s:').lower())
    preprocessed_sentence.do_replaces()
    preprocessed_sentence.stem_and_check_stop(stopword_set)
    preprocessed_sentence.make_tokens()
    preprocessed_sentence.make_original_text_tokens()
    preprocessed_sentence.text = ' '.join(sentence.tokens)
    return preprocessed_sentence


if __name__ == "__main__":
    app.run(host="127.0.0.1.", port=8000)