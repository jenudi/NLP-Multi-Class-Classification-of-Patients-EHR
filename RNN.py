import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+ hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size+ hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor,hidden_tensor),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)

'''
def init_rnn_old(rnn_model,criterion,optimizer,document,n_iters=100000):
    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    model=document.train.word2vec_for_rnn
    print("RNN with word2vec embeddings:")
    for i in range(n_iters):
        label, sentence,input_tensor, cls_numbers = document.train.make_random_sample_for_rnn(model)
        cls_numbers = torch.tensor(cls_numbers)
        cls_n = torch.reshape(cls_numbers, (-1,))
        output, loss = train_rnn(rnn_model,criterion,optimizer,input_tensor, cls_n)
        current_loss += loss
        print(f"Index: {i}, loss: {loss}")
        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0
        if (i + 1) % print_steps == 0:
            guess = document.train.labels_dict[int(torch.max(output, 1)[1].detach())]
            correct = "CORRECT" if guess == label else f"WRONG ({label})"
            print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {sentence} / {guess} {correct}")
    print("\n")
    plt.figure()
    plt.plot(all_losses)
    plt.show()
    torch.save(rnn_model.state_dict(), 'rnn_model.pth')


def train_rnn_old(rnn_model,criterion,optimizer, input_tensor, cls_numbers):
    rnn_model.train()
    hidden = rnn_model.init_hidden()
    output = None
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn_model(input_tensor[i], hidden)
    loss = criterion(output, cls_numbers)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()
  '''


def train_rnn(args, document, input_tensor, cls_numbers,rnn_model):
    criterion = nn.NLLLoss(weight=document.train.weights)
    optimizer = torch.optim.SGD(rnn_model.parameters(), lr=args.lr)
    rnn_model.train()
    hidden = rnn_model.init_hidden()
    output = None
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn_model(input_tensor[i], hidden)
    loss = criterion(output, cls_numbers)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


def init_rnn(args,document,model_name,n_iters=100000):
    current_loss = 0
    all_losses = list()
    plot_steps = 1000
    model=document.train.word2vec_for_rnn
    rnn_model = RNN(args.vec_size, args.hidden, len(document.train.labels_dict))
    print(f"model started: {model_name}")
    for iter in range(n_iters):
        label, input_tensor, cls_numbers = document.train.make_random_sample_for_rnn(model,model_name)
        cls_numbers = torch.tensor(cls_numbers)
        cls_n = torch.reshape(cls_numbers, (-1,))
        output, loss = train_rnn(args,document,input_tensor, cls_n,rnn_model)
        current_loss += loss
        print(f"index: {iter}, loss: {loss}")
        if (iter + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0
    '''
    plt.figure()
    plt.plot(all_losses)
    plt.show()
     '''
    return rnn_model

'''
def predict_rnn(document,rnn_model,predicted_set='validation'):
    model=document.train.word2vec_for_rnn
    if predicted_set=='validation':
        label, sentence,input_tensor,cls_numbers = document.validation.make_random_sample_for_rnn(model)
    else:
        label, sentence, input_tensor,cls_numbers = document.test.make_random_sample_for_rnn(model)
    print(f"Sentence tested with RNN of word2vec embeddings:\n {sentence}")
    with torch.no_grad():
        hidden = rnn_model.init_hidden()
        for i in range(input_tensor.size()[0]):
            output, hidden = rnn_model(input_tensor[i], hidden)
        guess = document.train.labels_dict[int(torch.max(output, 1)[1].detach())]
        print("Predicted label of sentence:\n" +guess+"\n")
  '''


def val_rnn(document,model_name,sentence_index,rnn_model,val=True):
    if val:
        tokens = document.validation.get_sentences_tokens()[sentence_index]
    else:
        tokens = document.test.get_sentences_tokens()[sentence_index]
    if model_name == 'w2v_p':
        input_tensor = torch.zeros(len(tokens), 1, 200)
    else:
        input_tensor = torch.zeros(len(tokens), 1, 300)
    for index, token in enumerate(tokens):
        try:
            numpy_copy = document.train.word2vec_for_rnn.wv[token].copy()
        except KeyError:
            if model_name == 'w2v_p':
                numpy_copy = np.zeros(200)
            else:
                numpy_copy = np.zeros(300)
        input_tensor[index][0][:] = torch.from_numpy(numpy_copy)
    with torch.no_grad():
        rnn_model.eval()
        hidden = rnn_model.init_hidden()
        for i in range(input_tensor.size()[0]):
            output, hidden = rnn_model(input_tensor[i], hidden)
        return int(torch.max(output, 1)[1].detach())


def eval_best_rnn_model(args,document):
    y_true = list()
    for sentence in document.validation.sentences:
        if sentence.label in document.train.labels_dict.values():
            position = list(document.train.labels_dict.values()).index(sentence.label)
            y_true.append(list(document.train.labels_dict.keys())[position])
        else:
            y_true.append(1000)
    check = pd.DataFrame(data={'y_true': y_true})
    models = ['w2v_3', 'w2v_5', 'w2v_p']
    learning_rates = [0.0005, 0.00001, 0.0001]
    hidden_layers = range(250, 400, 50)
    #train_tokens = args.doc.train.get_sentences_tokens()
    for model in models:
        for learning_rate in learning_rates:
            for hidden_layer in hidden_layers:
                args.hidden = hidden_layer
                args.lr = learning_rate
                if model == 'w2v_p':
                    document.train.make_word2vec_for_rnn(args,None)
                    args.vec_size = 200
                else:
                    if model == 'w2v_3':
                        document.train.make_word2vec_for_rnn(args,3)
                    if model == 'w2v_5':
                        document.train.make_word2vec_for_rnn(args,5)
                rnn_model = init_rnn(args,document,model,n_iters=100000)
                check[f'y_pred {model} {learning_rate} {hidden_layer}'] = [val_rnn(document,model,val_sentence_index,rnn_model,val=True) for val_sentence_index in range(len(document.validation.sentences))]
                print(check)
    print('Finished evaluating RNN models')