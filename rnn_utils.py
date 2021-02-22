import random
from gensim.models.word2vec import Word2Vec
import torch

'''
def make_embbedings(args):
    train_tokens = args.doc.train.get_sentences_tokens()
    word2vec_model = Word2Vec(min_count=args.min, window=5, size=args.vec_size,
                              sample=1e-3,alpha=0.03,min_alpha=0.0007)
    word2vec_model.build_vocab(train_tokens)
    word2vec_model.train(train_tokens, total_examples=word2vec_model.corpus_count, epochs=30)
    return word2vec_model


def make_random_sample(args,model):
    index = random.randrange(0, len(args.doc.train.sentences))
    tokens = args.doc.train.get_sentences_tokens()[index]
    label = args.doc.train.sentences[index].label
    sentence = args.doc.train.get_original_sentences()[index]
    input_tensor = torch.zeros(len(tokens), 1, 300)
    position = list(args.doc.train.labels_dict.values()).index(label)
    for i, v in enumerate(tokens):
        numpy_copy = model.wv[v].copy()
        input_tensor[i][0][:] = torch.from_numpy(numpy_copy)
    return label, sentence,input_tensor, list(args.doc.train.labels_dict.keys())[position]



def make_random_sample_val(args,model):
    index = random.randrange(0, len(args.doc.validation.sentences))
    tokens = args.doc.validation.get_sentences_tokens()[index]
    label = args.doc.validation.sentences[index].label
    sentence = args.doc.validation.get_original_sentences()[index]
    input_tensor = torch.zeros(len(tokens), 1, 300)
    #position = list(args.doc.val.labels_dict.values()).index(label)
    for i, v in enumerate(tokens):
        numpy_copy = model.wv[v].copy()
        input_tensor[i][0][:] = torch.from_numpy(numpy_copy)
    return label, sentence,input_tensor #, list(args.doc.val.labels_dict.keys())[position]

'''

