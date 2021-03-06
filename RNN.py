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


def train(rnn_model,optimizer,input_tensor, cls_numbers):
    criterion = nn.NLLLoss()
    hidden = rnn_model.init_hidden()
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn_model(input_tensor[i], hidden)
    loss = criterion(output, cls_numbers)
    l2_lambda = 0.01
    l2_norm = sum(p.pow(2.0).sum() for p in rnn_model.parameters())
    loss = loss + l2_lambda * l2_norm
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


def check_model(args,rnn_model,optimizer,document,n_iters):
    all_losses,val_losses = list(), list()
    model = document.train.word2vec_for_rnn
    model_name = document.train.word2vec_model_name
    current_loss,plot_steps = 0,5000
    for i in range(n_iters):
        label, input_tensor, cls_numbers, _ = document.train.make_random_sample_for_rnn(model,model_name,
                                                                                        document.labels_dict,
                                                                                        document.cls_numbers)
        cls_numbers = torch.tensor(cls_numbers)
        cls_n = torch.reshape(cls_numbers, (-1,))
        output, loss = train(rnn_model,optimizer,input_tensor, cls_n)
        current_loss += loss
        if (i + 1) % plot_steps == 0:
            print(f"index: {i+1}, loss: {loss:.4f}")
            all_losses.append(current_loss / plot_steps)
            current_loss = 0
            _, val_l = predict(rnn_model, document)
            val_losses.append(val_l / len(document.validation.sentences))

    plt.figure()
    plt.plot(all_losses, 'r', label="Train")
    plt.plot(val_losses, 'b', label="Validation")
    plt.suptitle(f"Model {model_name} lr: {args.lr} hidden layer: {args.hidden_layer}")
    plt.legend(loc="upper left")
    plt.xlabel('trained samples in 5K')  # fontsize=18
    plt.ylabel('loss')
    plt.show()
    torch.save(rnn_model.state_dict(), f'{document.train.word2vec_model_name}_rnn_model.pth')


def predict(rnn_model,document):
    criterion = nn.NLLLoss()
    temp_list = list()
    model = document.train.word2vec_for_rnn
    model_name = document.train.word2vec_model_name
    val_loss = 0.0
    for val_index in range(len(document.validation.sentences)):
        input_tup = document.validation.make_random_sample_for_rnn(model,model_name,document.labels_dict,document.cls_numbers,
                                                                   val_index)
        cls_ = torch.tensor(input_tup[2])
        cls_n = torch.reshape(cls_, (-1,))
        with torch.no_grad():
            hidden = rnn_model.init_hidden()
            for i in range(input_tup[1].size()[0]):
                output, hidden = rnn_model(input_tup[1][i], hidden)
            loss = criterion(output, cls_n)
            val_loss += loss.item()
            temp_list.append(int(torch.max(output, 1)[1].detach()))
    return temp_list, val_loss


def train_rnn_model(args,rnn_model,optimizer,document,n_iters=100000):
    check_model(args,rnn_model, optimizer, document, n_iters)
    y_pred, _ = predict(rnn_model,document)
    return y_pred

