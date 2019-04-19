import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.nn.init as init

from layers import ConvolutionalMultiheadSelfAttention as CMSA
from layers import ConvolutionalMultiheadTargetAttention as CMTA

class Proto(nn.Module):
    def __init__(self, num_emb, input_dim, pretrained_weight):
        super(Proto, self).__init__()
        self.id2vec = nn.Embedding(num_emb, input_dim, padding_idx=1)
        # unk, pad, ..., keywords
        self.id2vec.weight.data[3:].copy_(torch.from_numpy(pretrained_weight))
        self.id2vec.requires_grad = True

        # withLogits : combines a Sigmoid and BCE
        #self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()

    def binary_accuracy(self, y_pred, y):
        ge = torch.ge(y_pred.type(y.type()), 0.5).float()
        correct = torch.eq(ge, y).view(-1)

        return torch.sum(correct).item()/correct.shape[0]

    def accuracy(self, y_pred, y):
        _, pred = y_pred.max(1)
        return sum(pred==y).cpu().numpy()/y.size()

    def predict(self, x, l):
        input = self.id2vec(x)
        input = torch.div(torch.sum(input, 1), l)
        return self.model(input)

    def forward(self, data, sent_maxlen):
        x, l, y = torch.split(data, [sent_maxlen,1,1], 1)
        logits = self.predict(x, l.float())
        loss = self.loss(logits, y.squeeze())
        accuracy = self.accuracy(logits, y.squeeze())

        return loss, accuracy


class Proto_CNN(Proto):
    def __init__(self, input_dim, hidden_dim, kernel_dim,
                 sent_maxlen, dropout_rate, num_emb, pretrained_weight):
        super(Proto_CNN, self).__init__(num_emb, input_dim, pretrained_weight)

        self.positions = nn.Parameter(torch.randn(sent_maxlen,input_dim))
        stdv = 1. / self.positions.size(1) ** 0.5
        self.positions.data.uniform_(-stdv, stdv)
        self.cmsa = CMSA(input_dim, kernel_dim[0])
        self.cmta = CMTA(input_dim, kernel_dim[1])
        self.dropout = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(input_dim, hidden_dim[-1])
        nn.init.xavier_normal_(self.cls.weight)

    def predict(self, x, l):
        input = self.id2vec(x)
        input = self.dropout(input + self.positions)

        hidden = self.cmsa(input.permute(0,2,1))
        hidden = self.cmta(hidden)

        logits = self.cls(hidden.squeeze(-1))

        return logits
