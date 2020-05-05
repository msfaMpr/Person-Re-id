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

        self.hiddenDim = self.feature_dim // 2

        self.lstm = nn.LSTM(self.feature_dim, self.hiddenDim, bidirectional=True)
        # self.lstm_linear = nn.Linear(self.hiddenDim, self.hiddenDim)

        self.classifier = ClassBlock(self.part*self.feature_dim, self.class_num,
                                     droprate=0.5, relu=False, bnorm=True, num_bottleneck=256)

        # for i in range(self.part):
        #     name = 'classifierA'+str(i)
        #     setattr(self, name, ClassBlock(2*self.hiddenDim, self.class_num,
        #                                    droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-1):
        #     name = 'classifierB'+str(i)
        #     setattr(self, name, ClassBlock(4*self.hiddenDim, self.class_num,
        #                                    droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-1):
        #     name = 'classifierB'+str(i+self.part-1)
        #     setattr(self, name, ClassBlock(4*self.hiddenDim, self.class_num,
        #                                    droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-2):
        #     name = 'classifierC'+str(i)
        #     setattr(self, name, ClassBlock(6*self.hiddenDim, self.class_num,
        #                                    droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-2):
        #     name = 'classifierC'+str(i+self.part-2)
        #     setattr(self, name, ClassBlock(6*self.hiddenDim, self.class_num,
        #                                    droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-3):
        #     name = 'classifierD'+str(i)
        #     setattr(self, name, ClassBlock(8*self.hiddenDim, self.class_num,
        #                                    droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

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

        # partA, partB, partC, partD = {}, {}, {}, {}
        # predictA, predictB, predictC, predictD = {}, {}, {}, {}
        # y = []
        # # get six part feature batchsize*1280*4

        # for i in range(self.part):
        #     partA[i] = torch.flatten(x[:, i:i+1, :], 1)
        #     name = 'classifierA'+str(i)
        #     c = getattr(self, name)
        #     predictA[i] = c(partA[i])
        #     y.append(predictA[i])

        # for i in range(self.part-1):
        #     partB[i] = torch.flatten(x[:, i:i+2, :], 1)
        #     name = 'classifierB'+str(i)
        #     c = getattr(self, name)
        #     predictB[i] = c(partB[i])
        #     y.append(predictB[i])

        # for i in range(self.part-2):
        #     partC[i] = torch.flatten(x[:, i:i+3, :], 1)
        #     name = 'classifierC'+str(i)
        #     c = getattr(self, name)
        #     predictC[i] = c(partC[i])
        #     y.append(predictC[i])

        # for i in range(self.part-3):
        #     partD[i] = torch.flatten(x[:, i:i+4, :], 1)
        #     name = 'classifierD'+str(i)
        #     c = getattr(self, name)
        #     predictD[i] = c(partD[i])
        #     y.append(predictD[i])

        # partB[3] = torch.flatten(torch.cat((x[:, :1, :], x[:, 2:3, :]), 1), 1)
        # predictB[3] = self.classifierB3(partB[3])
        # y.append(predictB[3])

        # partB[4] = torch.flatten(torch.cat((x[:, :1, :], x[:, 3:4, :]), 1), 1)
        # predictB[4] = self.classifierB4(partB[4])
        # y.append(predictB[4])

        # partB[5] = torch.flatten(torch.cat((x[:, 1:2, :], x[:, 3:4, :]), 1), 1)
        # predictB[5] = self.classifierB5(partB[5])
        # y.append(predictB[5])

        # partC[2] = torch.flatten(torch.cat((x[:, :2, :], x[:, 3:4, :]), 1), 1)
        # predictC[2] = self.classifierC2(partC[2])
        # y.append(predictC[2])

        # partC[3] = torch.flatten(torch.cat((x[:, :1, :], x[:, 2:, :]), 1), 1)
        # predictC[3] = self.classifierC3(partC[3])
        # y.append(predictC[3])

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
        x = x.transpose(2, 1)  # bxhxp

        return x
