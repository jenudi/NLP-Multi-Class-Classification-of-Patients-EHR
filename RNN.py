import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+ hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size+ hidden_size,output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor,hidden_tensor),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.dropout(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)


class RecordsDataset(Dataset):
    def __init__(self, dataset, doc):
        self.dataset = dataset
        self.doc = doc
        self.model = doc.train.word2vec_for_rnn
        self.model_name = doc.train.word2vec_model_name
        self.dict = doc.labels_dict

    def __getitem__(self, index):
        label = self.dataset.sentences[index].label
        sentence = self.dataset.get_original_sentences()[index]
        if self.model_name == 'w2v_p':
            tokens = self.dataset.get_original_text_sentences_tokens()[index]
            input_tensor = torch.zeros(len(tokens), 1, 200)
        else:
            tokens = self.dataset.get_sentences_tokens()[index]
            input_tensor = torch.zeros(len(tokens), 1, 300)
        position = list(self.dict.values()).index(label)
        for i, v in enumerate(tokens):
            try:
                numpy_copy = self.model.wv[v].copy()
            except KeyError:
                if self.model_name == 'w2v_p':
                    numpy_copy = np.zeros(200)
                else:
                    numpy_copy = np.zeros(300)
            input_tensor[i][0][:] = torch.from_numpy(numpy_copy)
        return input_tensor, list(self.dict.keys())[position]

    def __len__(self):
        return len(self.dataset.sentences)


class TrainValidate:
    def __init__(self, args,document,rnn_model):
        self.args = args
        self.document = document
        self.rnn_model = rnn_model
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.optimizer = self.optimizer()
        self.class_weights = document.weights
        self.training_loss = 0.0
        self.val_loss = 0.0
        self.num_workers = 0
        self.all_training_loss = list()
        self.all_val_loss = list()
        self.y_pred = list()

    def optimizer(self):
        return torch.optim.SGD(self.rnn_model.parameters(), lr=self.args.lr,weight_decay=self.args.l2)

    def init_dls(self):
        train_data_set = RecordsDataset(dataset=self.document.train, doc=self.document)
        train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=True)
        val_data_set = RecordsDataset(dataset=self.document.validation, doc=self.document)
        val_data_loader = DataLoader(val_data_set, batch_size=1, shuffle=False)
        return train_data_loader,val_data_loader

    def lr_schedule(self,epoch):
      if 4 < epoch <= 7:
        self.args.lr = 0.00005
      elif 7 < epoch <= 11:
        self.args.lr = 0.000025
      elif epoch > 11:
        self.args.lr = 0.0000125

    def main(self, continue_training=False, decay_learning=False):
        if continue_training:
            try:
                self.rnn_model.load_state_dict(torch.load('rnn_model.pth'))
                print('Loaded model for training')
            except FileNotFoundError:
                print('File Not Found')
                return
        else:
            print('start training and validating')
        train_dl, val_dl = self.init_dls()
        self.clac_param()
        for epoch_ndx in range(self.args.epoch_num):
            print(f"epoch: {epoch_ndx}")
            if decay_learning:
                self.lr_schedule(epoch_ndx)
            trn_loss = self.training(epoch_ndx, train_dl)
            self.all_training_loss.append(trn_loss/len(train_dl))
            val_loss = self.validation(epoch_ndx, val_dl)
            self.all_val_loss.append(val_loss / len(val_dl))
        plt.figure()
        plt.plot(self.all_training_loss, 'r', label="Train")
        plt.plot(self.all_val_loss, 'b', label="Validation")
        plt.suptitle(f"Model {self.document.train.word2vec_model_name} "
                     f"lr: {self.args.lr} hidden layer: {self.args.hidden_layer}")
        plt.legend(loc="upper right")
        plt.xlabel('epochs')  # fontsize=18
        plt.ylabel('loss')
        plt.show()
        torch.save(self.rnn_model.state_dict(), f'{self.document.train.word2vec_model_name}_rnn_model.pth')
        print("finish training")
        return self.y_pred

    def training(self, epoch_ndx, train_dl):
        self.rnn_model.train()
        self.training_loss = 0.0
        for batch_ndx, batch_tup in enumerate(train_dl, 0):
            self.optimizer.zero_grad()
            loss,_ = self.compute_loss(batch_ndx, batch_tup, train_dl.batch_size)
            self.training_loss += loss
            loss.backward()
            self.optimizer.step()
        return self.training_loss

    def validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.rnn_model.eval()
            self.val_loss = 0.0
            for batch_ndx, batch_tup in enumerate(val_dl, 0):
                loss, output = self.compute_loss(batch_ndx, batch_tup, val_dl.batch_size)
                self.val_loss += loss
                if (epoch_ndx + 1) % self.args.epoch_num == 0:
                    self.y_pred.append(int(torch.max(output, 1)[1].detach()))
        return self.val_loss

    def compute_loss(self, batch_ndx, batch_tup, batch_size):
        loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        input, target = batch_tup
        target = torch.reshape(target, (-1,))
        hidden = self.rnn_model.init_hidden()
        for i in range(input.size(1)):
            output, hidden = self.rnn_model(input[0][i], hidden)
        loss = loss_func(output, target)
        return loss, output

    def clac_param(self):
        print(f"total parameters: {sum(p.numel() for p in self.rnn_model.parameters())}")
        print(f"trainable parameters: {sum(p.numel() for p in self.rnn_model.parameters() if p.requires_grad)}")