import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class ConvolutionalMultiheadAttention(Module):
    def __init__(self, input_dim, kernel_dim, multihead_cnt, conv_cnt):
        super(ConvolutionalMultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.multihead_cnt = multihead_cnt

        self.convs = nn.ModuleList([nn.Conv1d(input_dim, input_dim, kernel_dim)
                                    for _ in range(conv_cnt)])
        for w in self.convs:
            nn.init.xavier_normal_(w.weight)

    def attention(self, q, k, v):
        return torch.softmax(torch.div(torch.bmm(q.permute(0,2,1), k),
               np.sqrt(self.input_dim)), 2).bmm(v.permute(0,2,1)).permute(0,2,1)

    def multihead(self, hiddens):
        hiddens = [torch.chunk(hidden, self.multihead_cnt, 1) for hidden in hiddens]
        hiddens = torch.cat([self.attention(hiddens[0][i], hiddens[1][i], hiddens[2][i])
                             for i in range(self.multihead_cnt)], 1)

        return hiddens

class ConvolutionalMultiheadSelfAttention(ConvolutionalMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvolutionalMultiheadSelfAttention, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)

    def forward(self, input):
        hiddens = [F.elu(conv(input)) for conv in self.convs[:-1]]
        hiddens.append(torch.tanh(self.convs[-1](input)))

        elu_hid = self.multihead(hiddens[:3])
        tanh_hid= self.multihead(hiddens[3:])
        output = F.layer_norm(torch.mul(elu_hid, tanh_hid), elu_hid.size()[1:])

        return output

class ConvolutionalMultiheadTargetAttention(ConvolutionalMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=2):
        super(ConvolutionalMultiheadTargetAttention, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)
        self.target = nn.Parameter(torch.randn(input_dim, 1))
        stdv = 1. / math.sqrt(self.target.size(1))
        self.target.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.size(0)
        hiddens = [F.elu(conv(input)) for conv in self.convs]
        output = self.multihead([self.target.expand(batch_size, self.input_dim, 1)]+hiddens)

        return output
