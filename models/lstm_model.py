import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from .base_model import *


class PCB_Effi_LSTM(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_LSTM, self).__init__()

        self.class_num = model.class_num
        self.part = model.part
        self.model = model.model
        self.avgpool = model.avgpool
        self.dropout = nn.Dropout(p=0.5)
        self.feature_dim = model.feature_dim

        self.hiddenDim = self.feature_dim // self.part
        # self.hiddenDim = self.feature_dim

        self.lstm = nn.LSTM(self.feature_dim, self.hiddenDim, bidirectional=True)
        # self.lstm_linear = nn.Linear(self.hiddenDim, self.hiddenDim)

        self.classifier = ClassBlock(
            2*self.feature_dim, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.extract_features(x)
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.squeeze()

        batchSize, seq_len = x.size(0), x.size(2)

        # h0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()
        # c0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()

        x = x.transpose(2, 1)  # bxpx1280
        x = x.transpose(1, 0)  # pxbx1280

        # output, hn = self.lstm(x, (h0, c0))
        output, hn = self.lstm(x)

        # x = output.reshape((output.size(0) * output.size(1), self.hiddenDim))
        # x = self.lstm_linear(x)
        # x = x.reshape((output.size(0), output.size(1), self.hiddenDim))
        x = output.transpose(1, 0)  # bxpxh
        # x = x.transpose(2, 1) # bxhxp

        x = torch.flatten(x, 1)
        y = self.classifier(x)

        return y


class PCB_Effi_LSTM_test(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_LSTM_test, self).__init__()
        self.part = model.part
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

        self.hiddenDim = model.hiddenDim
        self.lstm = model.lstm
        # self.lstm_linear = model.lstm_linear

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.avgpool(x)
        x = x.squeeze()

        batchSize, seq_len = x.size(0), x.size(2)

        # h0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()
        # c0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()

        x = x.transpose(2, 1)  # bxpx1280
        x = x.transpose(1, 0)  # pxbx1280

        # output, hn = self.lstm(x, (h0, c0))
        output, hn = self.lstm(x)

        # x = output.reshape((output.size(0) * output.size(1), self.hiddenDim))
        # x = self.lstm_linear(x)
        # x = x.reshape((output.size(0), output.size(1), self.hiddenDim))
        x = output.transpose(1, 0)  # bxpxh
        x = x.transpose(2, 1) # bxhxp

        return x
