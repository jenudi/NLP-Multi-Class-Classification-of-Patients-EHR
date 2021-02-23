import torch
import torch.nn as nn
from matplotlib import pyplot as plt


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


def init_rnn(rnn_model,criterion,optimizer,document,n_iters=100000):
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
    '''
    plt.figure()
    plt.plot(all_losses)
    plt.show()
    '''
    torch.save(rnn_model.state_dict(), 'rnn_model.pth')


def train_rnn(rnn_model,criterion,optimizer, input_tensor, cls_numbers):
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