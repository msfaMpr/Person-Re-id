import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class PCB(nn.Module):
    def __init__(self, opt):
        super(PCB, self).__init__()
        self.part = opt.nparts
        self.class_num = opt.nclasses
        self.single_cls = opt.single_cls
        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # self.dropout = nn.Dropout(p=0.5)

        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        self.feature_dim = self.model.fc.in_features

        if self.single_cls:
            name = 'classifier'
            setattr(self, name, ClassBlock(self.part*self.feature_dim, self.class_num,
                                        droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
        else:
            for i in range(self.part):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(self.feature_dim, self.class_num,
                                            droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        # x = self.dropout(x)

        part = {}
        predict = {}
        y = []

        if self.single_cls:
            part[0] = torch.squeeze(x[:, :, 0])
            name = 'classifier'
            c = getattr(self, name)
            predict[0] = c(part[0])
            y = predict[0]
        else:
            for i in range(self.part):
                part[i] = torch.squeeze(x[:, :, i])
                name = 'classifier'+str(i)
                c = getattr(self, name)
                predict[i] = c(part[i])
            for i in range(self.part):
                y.append(predict[i])

        return y, part[0]


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = model.part
        self.model = model.model
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class PCB_Effi(nn.Module):
    def __init__(self, opt):
        super(PCB_Effi, self).__init__()
        self.class_num = opt.nclasses
        self.part = opt.nparts
        self.single_cls = opt.single_cls
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.feature_dim = self.model._fc.in_features

        if self.single_cls:
            name = 'classifier'
            setattr(self, name, ClassBlock(self.part*self.feature_dim, self.class_num,
                                        droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
        else:
            for i in range(self.part):
                name = 'classifierA'+str(i)
                setattr(self, name, ClassBlock(self.feature_dim, self.class_num, droprate=0.5,
                                            relu=False, bnorm=True, num_bottleneck=128))

            for i in range(self.part-1):
                name = 'classifierB'+str(i)
                setattr(self, name, ClassBlock(2*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            for i in range(self.part-1):
                name = 'classifierB'+str(i+self.part-1)
                setattr(self, name, ClassBlock(2*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            for i in range(self.part-2):

                name = 'classifierC'+str(i)
                setattr(self, name, ClassBlock(3*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            for i in range(self.part-2):
                name = 'classifierC'+str(i+self.part-2)
                setattr(self, name, ClassBlock(3*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            for i in range(self.part-3):
                name = 'classifierD'+str(i)
                setattr(self, name, ClassBlock(4*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.transpose(x, 1, 2).squeeze()
        
        partA, partB, partC, partD = {}, {}, {}, {}
        predictA, predictB, predictC, predictD = {}, {}, {}, {}
        
        y = {}
        y['PCB'] = []

        if self.single_cls:
            name = 'classifier'
            c = getattr(self, name)
            y = c(x)
        else:
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


class PCB_Effi_test(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_test, self).__init__()
        self.part = model.part
        self.model = model.model
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y
