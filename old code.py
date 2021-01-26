#corpus = {}
#for i, sentence in enumerate(tokens):
#    corpus['sent' + str(i)] = Counter(sentence)
#df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

'''
#%% DBSCAN
epsilons=[0.1:0.1:1]
clustering = DBSCAN(eps=3, min_samples=50).fit(doc.train.tfidf.todense())
print("number of groups: " + str())
'''

'''
#%% SVD
## maybe change to t-SNE
U, s, Vt = np.linalg.svd(tfs_dataframe.T)
S = np.zeros((len(U), len(Vt)))
pd.np.fill_diagonal(S, s)
pd.DataFrame(S).round(1)

#%% SVD err
err = []
for numdim in range(len(s), 0, -1):
    S[numdim - 1, numdim - 1] = 0
    tfs_dataframe_reconstructed = U.dot(S).dot(Vt)
    err.append(np.sqrt(((tfs_dataframe_reconstructed - tfs_dataframe.T).values.flatten() ** 2).sum() / np.product(tfs_dataframe.T.shape)))
np.array(err).round(2)

ts = ts.cumsum()
ts.plot()
plt.show()
'''

"""
#%% TF-IDF Manual
document_tfidf_vectors = []
for sent in sentences:
    vec = copy.copy(zero_vector)
    tokens = sent
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        sents_containing_key = 0
        for _sent in sentences:
            if key in  _sent:
                sents_containing_key += 1
        tf = value / len(lexicon)
        if sents_containing_key:
            #idf = len(sentences) / sents_containing_key
            idf = (1+np.log((1+len(sentences)) / (1+sents_containing_key)))
        else:
            idf = 0
        vec[key] = tf * idf
    document_tfidf_vectors.append(vec)

corpus = {}
for i, n in enumerate(document_tfidf_vectors):
    corpus['sent' + str(i)] = n
df = pd.DataFrame.from_records(corpus).T
"""

#%%
#sentences = [re.split(r'[-\s.,;!?]+', i) for i in temp_sentences]
#for i, token in enumerate(sentences):
 #  sentences[i] = [w for w in token if not w in stopword_set]

#%% n-grams
# One
#sentences = [re.split(r'[-\s.,;!?]+', i) for i in temp_sentences]

#corpus = {}
#for i, sentence in enumerate(sentences):
 #   corpus['sent' + str(i)] = dict((token, 1) for token in sentence)
#df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

# Two-grams
#tokens_two = list(ngrams(i, 2) for i in sentences)

sent_train = doc.train.get_original_sentences()

'''
newl = list()
for i,v in enumerate(doc.train.get_original_sentences()):
    a = np.zeros(200)
    for j in v.split():
        if j not in word2vec_pubmed_model.vocab.keys():
            a += np.zeros(200)
        else:
            a += word2vec_pubmed_model.wv[j]
    newl.append(a)

newl_val = list()
for i,v in enumerate(doc.validation.get_original_sentences()):
    a = np.zeros(200)
    for j in v.split():
        if j not in word_vectors.vocab.keys():
            a += np.zeros(200)
        else:
            a += word_vectors.wv[j]
    newl_val.append(a)
'''
#if some words in our lexicon dont exist in words2vec lexicon change them to 'unknown' token
#check how to determine number of features

# Word2Vec Hyper-parameters:
# 1.Number of tokens
# 2. alpha - learning rate
# 3. window
# 4. Size of vector
# 5. Compare it to other models

# After the training one should load the saved model, so the following code should not executed again:

#word2vec_model.save("word2vec.model")

#word2vec_model = Word2Vec.load("word2vec.model")

#word_vectors = KeyedVectors.load_word2vec_format('../RANZCR/PubMed-w2v.bin', binary=True)
#word_vectors.save("word2vec_pubmed.model")

#word2vec_model = Word2Vec.load("word2vec.model")