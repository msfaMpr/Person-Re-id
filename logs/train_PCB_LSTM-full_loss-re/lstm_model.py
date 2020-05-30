import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from .base_model import ClassBlock


class PCB_Effi_LSTM(nn.Module):
    def __init__(self, model, train_backbone=False):
        super(PCB_Effi_LSTM, self).__init__()

        self.train_backbone = train_backbone
        self.class_num = model.class_num
        self.part = model.part
        self.model = model.model
        self.avgpool = model.avgpool
        self.dropout = nn.Dropout(p=0.5)
        self.feature_dim = model.feature_dim

        # self.glob_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.glob_dropout = nn.Dropout(p=0.5)

        self.hiddenDim = self.feature_dim // 2
        self.lstm = nn.LSTM(
            self.feature_dim, self.hiddenDim, bidirectional=True)

        self.classifier = ClassBlock(self.part * self.feature_dim, self.class_num,
                                     droprate=0.5, relu=False, bnorm=True, num_bottleneck=256)

        for i in range(self.part):
            name = 'classifierA'+str(i)
            setattr(self, name, ClassBlock(2 * self.hiddenDim, self.class_num,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        for i in range(self.part-1):
            name = 'classifierB'+str(i)
            setattr(self, name, ClassBlock(4*self.hiddenDim, self.class_num,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        for i in range(self.part-1):
            name = 'classifierB'+str(i+self.part-1)
            setattr(self, name, ClassBlock(4*self.hiddenDim, self.class_num,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        for i in range(self.part-2):
            name = 'classifierC'+str(i)
            setattr(self, name, ClassBlock(6*self.hiddenDim, self.class_num,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        for i in range(self.part-2):
            name = 'classifierC'+str(i+self.part-2)
            setattr(self, name, ClassBlock(6*self.hiddenDim, self.class_num,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        for i in range(self.part-3):
            name = 'classifierD'+str(i)
            setattr(self, name, ClassBlock(8*self.hiddenDim, self.class_num,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        if self.train_backbone:
            x = self.model.extract_features(x)
        else:
            with torch.no_grad():
                x = self.model.extract_features(x)

        # gx = self.glob_avgpool(x)
        # gx = self.glob_dropout(gx)
        # gx = gx.squeeze()

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.squeeze()

        batchSize, seq_len = x.size(0), x.size(2)

        h0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()
        # h0 = gx.view(2, gx.size(0), gx.size(1) // 2)
        c0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()
        # c0 = gx.view(2, gx.size(0), gx.size(1) // 2)

        x = x.transpose(2, 1)  # bxpx1280

        lx = x.transpose(1, 0)  # pxbx1280
        lx, hn = self.lstm(lx, (h0, c0))
        lx = lx.transpose(1, 0)  # bxpxh
        lx = torch.flatten(lx, 1)

        y = {}
        y['LSTM'] = self.classifier(lx)

        partA, partB, partC, partD = {}, {}, {}, {}
        predictA, predictB, predictC, predictD = {}, {}, {}, {}
        y['PCB'] = []
        # get six part feature batchsize*1280*4

        for i in range(self.part):
            partA[i] = torch.flatten(x[:, i:i+1, :], 1)
            name = 'classifierA'+str(i)
            c = getattr(self, name)
            predictA[i] = c(partA[i])
            y['PCB'].append(predictA[i])

        for i in range(self.part-1):
            partB[i] = torch.flatten(x[:, i:i+2, :], 1)
            name = 'classifierB'+str(i)
            c = getattr(self, name)
            predictB[i] = c(partB[i])
            y['PCB'].append(predictB[i])

        for i in range(self.part-2):
            partC[i] = torch.flatten(x[:, i:i+3, :], 1)
            name = 'classifierC'+str(i)
            c = getattr(self, name)
            predictC[i] = c(partC[i])
            y['PCB'].append(predictC[i])

        for i in range(self.part-3):
            partD[i] = torch.flatten(x[:, i:i+4, :], 1)
            name = 'classifierD'+str(i)
            c = getattr(self, name)
            predictD[i] = c(partD[i])
            y['PCB'].append(predictD[i])

        partB[3] = torch.flatten(torch.cat((x[:, :1, :], x[:, 2:3, :]), 1), 1)
        predictB[3] = self.classifierB3(partB[3])
        y['PCB'].append(predictB[3])

        partB[4] = torch.flatten(torch.cat((x[:, :1, :], x[:, 3:4, :]), 1), 1)
        predictB[4] = self.classifierB4(partB[4])
        y['PCB'].append(predictB[4])

        partB[5] = torch.flatten(torch.cat((x[:, 1:2, :], x[:, 3:4, :]), 1), 1)
        predictB[5] = self.classifierB5(partB[5])
        y['PCB'].append(predictB[5])

        partC[2] = torch.flatten(torch.cat((x[:, :2, :], x[:, 3:4, :]), 1), 1)
        predictC[2] = self.classifierC2(partC[2])
        y['PCB'].append(predictC[2])

        partC[3] = torch.flatten(torch.cat((x[:, :1, :], x[:, 2:, :]), 1), 1)
        predictC[3] = self.classifierC3(partC[3])
        y['PCB'].append(predictC[3])

        return y


class PCB_Effi_LSTM_test(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_LSTM_test, self).__init__()
        self.part = model.part
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

        # self.glob_avgpool = model.glob_avgpool

        self.hiddenDim = model.hiddenDim
        self.lstm = model.lstm

    def forward(self, x):
        x = self.model.extract_features(x)

        # gx = self.glob_avgpool(x)
        # gx = gx.squeeze()

        x = self.avgpool(x)
        x = x.squeeze()

        # batchSize, seq_len = x.size(0), x.size(2)
        #
        # h0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()
        # # h0 = gx.view(2, gx.size(0), gx.size(1) // 2)
        # c0 = Variable(torch.zeros(2, x.size(0), self.hiddenDim)).cuda()
        # # c0 = gx.view(2, gx.size(0), gx.size(1) // 2)
        #
        # x = x.transpose(2, 1)  # bxpx1280
        # x = x.transpose(1, 0)  # pxbx1280
        #
        # output, hn = self.lstm(x, (h0, c0))
        #
        # x = output.transpose(1, 0)  # bxpxh
        # x = x.transpose(2, 1)  # bxhxp

        return x
