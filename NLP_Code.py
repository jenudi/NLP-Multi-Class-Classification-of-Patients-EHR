# Intro to NLP with sci-kit learn (sklearn)
import os
import pandas as pd
import numpy as np
import pickle
from warnings import warn
from typing import Sequence, Any, Set, Pattern as RePattern

import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from itertools import chain, cycle
from datetime import datetime as dt
from collections import Counter
from multiprocessing import cpu_count

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, \
    GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, roc_auc_score, recall_score, \
    accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
    roc_curve, auc

# New imports
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import json

try:
    _ = WordNetLemmatizer().lemmatize('loving', 'v')
except LookupError:
    import nltk
    nltk.download('wordnet')

try:
    _ = stopwords.words("english")
except LookupError:
    import nltk
    nltk.download('stopwords')

dataset_folder = r'data sets\20news-18828'
save_folder = r'cache\Search, Information Retrieval and Recommender Systems\NLP example'

#%%
precompiled_space_re = [re.compile(r"[^A-Za-z0-9^,!.\/'+-=]>"),  # unexpected chars
                        re.compile('<[\w\W]+>'),  # br tags
                        re.compile(r"\s+")]  # merge white space characters, including line breaks
stemmer = SnowballStemmer('english')
stopword_set = set(stopwords.words("english"))

def preprocess_document(document: str, stem_func: callable=None, \
                        precompiled_space_re: Sequence[RePattern]=None, \
                        stopword_set: Set[str]=set()):
    # Clean the text, with the option to remove stopwords and to stem words.

    document = document.lower()

    document = '\n'.join((line) for line in document.split('\n') \
                         if len(line) > 1 and not (line.startswith('from:') \
                                                   or line.startswith('subject:')))

    # Clean the text
    document = document.replace(r"'s", "  is")
    document = document.replace(r"'ve", " have")
    document = document.replace(r"can't", "cannot")
    document = document.replace(r"musn't", "must not")
    document = document.replace(r"n't", " not")
    document = document.replace(r"i'm", "i am")
    document = document.replace(r"'re", " are")
    document = document.replace(r"'d", " would")
    document = document.replace(r"\'ll", " will")
    document = document.replace(r",", " ")
    document = document.replace(r".", " . ")
    document = document.replace(r"!", " ! ")
    document = document.replace(r"/", " ")
    document = document.replace(r"^", " ^ ")
    document = document.replace(r"+", " + ")
    document = document.replace(r"-", " - ")
    document = document.replace(r"=", " = ")
    document = document.replace(r"'", " ")
    document = document.replace(r":", " : ")
    document = document.replace(r" e g ", " eg ")
    document = document.replace(r" b g ", " bg ")
    document = document.replace(r" u s ", " united states ")
    document = document.replace(r" 9 11 ", "911")
    document = document.replace(r"e - mail", "email")
    document = document.replace(r"e-mail", "email")
    document = document.replace(r" e mail", "email")
    document = document.replace(r"email", "email")
    document = document.replace(r"j k", "jk")
    document = document.replace(r"shoulda", "should have")
    document = document.replace(r"coulda", "could have")
    document = document.replace(r"woulda", "would have")
    document = document.replace(r"http", "")

    document = document.replace(r">", "> ")  # This is specific for 20 news groups

    if precompiled_space_re:
        for cur_re in precompiled_space_re:
            document = re.sub(cur_re, " ", document)

    document = document.split()
    # Remove stop words and stem words
    if stem_func:
        document = [stem_func(w) for w in document]

    if stopword_set:
        document = [w for w in document if not w in stopword_set]

    return ' '.join(document)

class_names =  os.listdir(dataset_folder)
labels = []
docs = []
file_names = []
for class_ in class_names:
    files = os.listdir(dataset_folder + os.sep + class_)
    labels.extend([class_]*len(files))
    file_names.extend([class_ + os.sep + file for file in files])
    for file in files:
        with open(os.sep.join((dataset_folder, class_, file)), 'r') as f:
            doc = f.read()
        docs.append(preprocess_document(doc, stemmer.stem, precompiled_space_re, stopword_set))
    if not len(docs) == len(labels):
        raise RuntimeError("The number of labels isn't equal to the number of documents")

# # Save the parsed dataset for future use
# with open(save_folder+os.sep+'parsed_dataset.json', 'w') as f:
#     json.dump((labels, docs), f)

#%% Matching to code from previous example
curated_df = pd.DataFrame(columns=['syscalls','label', 'filenames'])
curated_df['syscalls'] = docs
curated_df['label'] = labels
malware_names = file_names
labels = class_names
