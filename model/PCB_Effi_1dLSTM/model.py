import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
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

# Define the ResNet50-based Model


class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model


class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the NAS-based Model


class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_name = 'nasnetalarge'
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).


class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)


class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 4  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5,
                                           relu=False, bnorm=True, num_bottleneck=256))

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
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 4
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

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
    def __init__(self, class_num):
        super(PCB_Effi, self).__init__()

        self.class_num = class_num
        self.part = 4  # We cut the pool5 to 4 parts
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.feature_dim = 1280

        # define 4 classifiers
        for i in range(self.part):
            name = 'classifierA'+str(i)
            setattr(self, name, ClassBlock(self.feature_dim, self.class_num, droprate=0.5,
                                           relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-1):
        #     name = 'classifierB'+str(i)
        #     setattr(self, name, ClassBlock(2*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-1):
        #     name = 'classifierB'+str(i+self.part-1)
        #     setattr(self, name, ClassBlock(2*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-2):
        #     name = 'classifierC'+str(i)
        #     setattr(self, name, ClassBlock(3*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-2):
        #     name = 'classifierC'+str(i+self.part-2)
        #     setattr(self, name, ClassBlock(3*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        # for i in range(self.part-3):
        #     name = 'classifierD'+str(i)
        #     setattr(self, name, ClassBlock(4*1280, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.transpose(x, 1, 2).squeeze()

        partA, partB, partC, partD = {}, {}, {}, {}
        predictA, predictB, predictC, predictD = {}, {}, {}, {}
        y = []
        # get six part feature batchsize*1280*4

        for i in range(self.part):
            partA[i] = torch.flatten(x[:, i:i+1, :], 1)
            name = 'classifierA'+str(i)
            c = getattr(self, name)
            predictA[i] = c(partA[i])
            y.append(predictA[i])

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

        # sum prediction
        #y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]

        return y


class PCB_Effi_test(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_test, self).__init__()
        self.part = model.part
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


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

        self.lstm = nn.LSTM(self.feature_dim, self.hiddenDim)
        # self.lstm_linear = nn.Linear(self.hiddenDim, self.hiddenDim)

        self.classifier = ClassBlock(
            self.feature_dim, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.extract_features(x)
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.squeeze()

        batchSize, seq_len = x.size(0), x.size(2)

        h0 = Variable(torch.zeros(1, x.size(0), self.hiddenDim)).cuda()
        c0 = Variable(torch.zeros(1, x.size(0), self.hiddenDim)).cuda()

        x = x.transpose(2, 1)  # bxpx1280
        x = x.transpose(1, 0)  # pxbx1280

        output, hn = self.lstm(x, (h0, c0))
        # x = output.reshape((output.size(0) * output.size(1), self.hiddenDim))
        # x = self.lstm_linear(x)
        # x = x.reshape((output.size(0), output.size(1), self.hiddenDim))
        x = output.transpose(1, 0)  # bxpxh
        x = x.transpose(2, 1) # bxhxp
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
        self.lstm_linear = model.lstm_linear

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.avgpool(x)
        x = x.squeeze()

        batchSize, seq_len = x.size(0), x.size(2)

        h0 = Variable(torch.zeros(1, x.size(0), self.hiddenDim)).cuda()
        c0 = Variable(torch.zeros(1, x.size(0), self.hiddenDim)).cuda()

        x = x.transpose(2, 1)  # bxpx1280
        x = x.transpose(1, 0)  # pxbx1280

        output, hn = self.lstm(x, (h0, c0))
        x = output.reshape((output.size(0) * output.size(1), self.hiddenDim))
        # x = self.lstm_linear(x)
        x = x.reshape((output.size(0), output.size(1), self.hiddenDim))
        x = output.transpose(1, 0)  # bxpxh
        x = x.transpose(2, 1) # bxhxp

        return x


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)
