import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from layers import *


class DisGCN(nn.Module):
    def __init__(self, in_dim, channels, C_dim, iterations, beta, layer_num, dropout,
                 out_dim):  # 输入维数， 通道数目， 每个通道的输出维数， 迭代次数， 平衡因子
        super(DisGCN, self).__init__()
        self.dropout_rate = dropout
        self.channels = channels
        self.C_dim = C_dim
        self.iterations = iterations
        self.beta = beta
        self.layer_num = layer_num
        self.disconv1 = DisConv(in_dim[0], channels[0], C_dim[0], iterations, beta)
        self.disconv2 = DisConv(in_dim[1], channels[1], C_dim[1], iterations, beta)
        self.W_o = torch.nn.Parameter(torch.empty(size=(channels[-1] * C_dim[-1], out_dim), dtype=torch.float),
                                      requires_grad=True)
        self.bias = torch.nn.Parameter(torch.empty(size=(1, out_dim), dtype=torch.float), requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)

    def forward(self, adj, features):
        h = features
        h = self.disconv1(adj, h)
        h = F.dropout(h, self.dropout_rate, training=self.training)
        h = self.disconv2(adj, h)
        h = F.dropout(h, self.dropout_rate, training=self.training)
        output = torch.mm(h, self.W_o) + self.bias
        return output
