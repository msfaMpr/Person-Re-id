from __future__ import print_function, division

import time
import os
import argparse
import yaml
import math
from shutil import copyfile
#from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch

from datasets import init_dataset, ImageDataset, VideoDataset, RandomIdentitySampler
from utils.loss_triplet import TripletLoss, CrossEntropyLabelSmooth
from utils.loss_center import CenterLoss
from utils.random_erasing import RandomErasing

from models.vrid_model import VRidGGNN
from models.base_model import PCB, PCB_Effi


######################################################################
# Options
# --------
#

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='../Market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--nparts', default=4, type=int, help='number of stipes')
parser.add_argument('--erasing_p', default=0.0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--single_cls', action='store_true', help='use signle classifier')
parser.add_argument('--LSTM', action='store_true', help='use LSTM')
parser.add_argument('--GGNN', action='store_true', help='use GGNN')
parser.add_argument('--backbone', default='EfficientNet-B0', type=str, help='backbone model name')
parser.add_argument('--freeze_backbone', action='store_true', help='train backbone network')
parser.add_argument('--use_triplet_loss', action='store_true', help='use triplet loss for training')
parser.add_argument('--label_smoothing', action='store_true', help='use label smoothing')
parser.add_argument('--bidirectional', action='store_true', help='use bidirectional lstm')
parser.add_argument('--seq_len', default=4, type=int, help='number of frames in a sample')
parser.add_argument('--sample_method', default='random', type=str, help='method to sample frames')

opt = parser.parse_args()

# opt.use_triplet_loss = True
opt.label_smoothing = True
opt.bidirectional = opt.LSTM


######################################################################
# Set GPU
# --------
#

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


######################################################################
# Load Data
# --------
#

normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = T.Compose([
    T.Resize([384, 128]),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop([384, 128]),
    T.ToTensor(),
    normalize_transform,
    RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
])

# val_transforms = T.Compose([
#     T.Resize([384, 128]),
#     T.ToTensor(),
#     normalize_transform
# ])

dataset = init_dataset('mars', root='../')
dataset_sizes = {}
dataset_sizes['train'] = dataset.num_train_imgs
train_set = VideoDataset(dataset.train, opt.seq_len, opt.sample_method,train_transforms)
dataloaders = {}
dataloaders['train'] = DataLoader(
    train_set, batch_size=opt.batchsize, drop_last=True,
    sampler=RandomIdentitySampler(dataset.train, opt.batchsize, 4), num_workers=8)

# val_set = VideoDataset(dataset.query + dataset.gallery, 4, val_transforms)
# dataloaders['val'] = DataLoader(
#     val_set, batch_size=opt.batchsize, drop_last=True, shuffle=False, num_workers=8)

use_gpu = torch.cuda.is_available()


######################################################################
# Training the model
# --------
#

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, loss_func, optimizer, scheduler, num_epochs=25):
    since = time.time()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(
        dataset_sizes['train'] / (opt.batchsize * 4))*opt.warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:

            if phase == 'train':
                model.train(True)  # Set model to training mode
                if opt.freeze_backbone:
                    model.model.train(False)
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:

                # get the inputs
                if phase == 'train':
                    inputs, labels, _ = data
                else:
                    inputs, labels = data

                now_batch_size, _, _, _, _ = inputs.shape

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        if opt.use_triplet_loss:
                            outputs, features = model(inputs)
                        else:
                            outputs = model(inputs)
                            features = None
                else:
                    if opt.use_triplet_loss:
                        outputs, features = model(inputs)
                    else:
                        outputs = model(inputs)
                        features = None

                if opt.single_cls:
                    outputs = outputs['GGNN']
                    _, preds = torch.max(outputs.data, 1)
                    loss = loss_func(outputs, features, labels)
                else:
                    part = {}
                    feat = {}
                    sm = nn.Softmax(dim=1)
                    num_part = opt.nparts

                    for i in range(num_part):
                        part[i] = outputs['PCB'][i]

                    score = sm(part[0]) + sm(part[1]) + \
                        sm(part[2]) + sm(part[3])

                    _, preds = torch.max(score.data, 1)

                    r_labels = labels.view(-1, 1).repeat(1, 4).view(-1)
                    loss = loss_func(part[0], features, r_labels)
                    for i in range(1, num_part):
                        loss += loss_func(part[i], features, r_labels)

                    # for i in range(num_part-1):
                    #     loss += loss_func(outputs['PCB'][num_part+i], features, labels)

                    # for i in range(num_part-2):
                    #     loss + loss_func(outputs['PCB']
                    #                      [2*num_part+i-1], features, labels)

                    # for i in range(num_part-3):
                    #     loss + loss_func(outputs['PCB']
                    #                      [3*num_part+i-3], features, labels)

                    # for i in range(5):
                    #     loss += loss_func(outputs['PCB'][10+i], features, labels)

                    if opt.LSTM:
                        # loss /= 5.0
                        loss += loss_func(outputs['LSTM'], features, labels)
                    if opt.GGNN:
                        # loss /= 5.0
                        loss += loss_func(outputs['GGNN'], features, labels)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == r_labels.data))

            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)

            # deep copy the model
            if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch+1)
                draw_curve(epoch+1)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# --------
#

x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    # ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    # ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./logs', opt.name, 'train.jpg'))


######################################################################
# Save model
# --------
#

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./logs', opt.name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Load model
# --------
#

def load_network(network, model_name):
    save_path = os.path.join('./logs', model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Finetuning the convnet
# --------
#

opt.nclasses = dataset.num_train_pids

if opt.backbone == 'ResNet50':
    model = PCB(opt)
elif opt.backbone == 'EfficientNet-B0':
    model = PCB_Effi(opt)

# model_name = 'test-pcb-ac'
# model = load_network(model, model_name)
model = VRidGGNN(model)
# model_name = 'VRid'
# model = load_network(model, model_name)

print(model)

if opt.single_cls:
    if opt.backbone == 'ResNet50':
        ignored_params = list(map(id, model.model.fc.parameters()))
    elif opt.backbone == 'EfficientNet-B0':
        ignored_params = list(map(id, model.model._fc.parameters()))
    ignored_params += (list(map(id, model.classifier.parameters())))
    if opt.freeze_backbone:
        ignored_params += (list(map(id, model.model.parameters())))
    base_params = filter(
        lambda p: id(p) not in ignored_params, model.parameters()
    )
    # optimizer = optim.SGD(
    #     [{'params': base_params, 'lr': 0.1*opt.lr},
    #      {'params': model.classifier.parameters(), 'lr': opt.lr}],
    #     weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(
        [{'params': base_params, 'lr': 0.00035},
         {'params': model.classifier.parameters(), 'lr': 0.0035}]
    )
else:
    if opt.backbone == 'ResNet50':
        ignored_params = list(map(id, model.model.fc.parameters()))
    elif opt.backbone == 'EfficientNet-B0':
        ignored_params = list(map(id, model.model._fc.parameters()))
    ignored_params += (
        list(map(id, model.classifierA0.parameters()))
        + list(map(id, model.classifierA1.parameters()))
        + list(map(id, model.classifierA2.parameters()))
        + list(map(id, model.classifierA3.parameters()))

        # + list(map(id, model.classifierB0.parameters()))
        # + list(map(id, model.classifierB1.parameters()))
        # + list(map(id, model.classifierB2.parameters()))

        # + list(map(id, model.classifierC0.parameters()))
        # + list(map(id, model.classifierC1.parameters()))

        # + list(map(id, model.classifierD0.parameters()))

        # + list(map(id, model.classifierB3.parameters()))
        # + list(map(id, model.classifierB4.parameters()))
        # + list(map(id, model.classifierB5.parameters()))

        # + list(map(id, model.classifierC2.parameters()))
        # + list(map(id, model.classifierC3.parameters()))
        + list(map(id, model.classifier.parameters()))
    )
    if opt.freeze_backbone:
        ignored_params += (list(map(id, model.model.parameters())))
    base_params = filter(
        lambda p: id(p) not in ignored_params, model.parameters()
    )

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 0.00035},

        {'params': model.classifierA0.parameters(), 'lr': 0.0035},
        {'params': model.classifierA1.parameters(), 'lr': 0.0035},
        {'params': model.classifierA2.parameters(), 'lr': 0.0035},
        {'params': model.classifierA3.parameters(), 'lr': 0.0035},
        
        # {'params': model.classifierB0.parameters(), 'lr': 0.0035},
        # {'params': model.classifierB1.parameters(), 'lr': 0.0035},
        # {'params': model.classifierB2.parameters(), 'lr': 0.0035},

        # {'params': model.classifierC0.parameters(), 'lr': 0.0035},
        # {'params': model.classifierC1.parameters(), 'lr': 0.0035},

        # {'params': model.classifierD0.parameters(), 'lr': 0.0035},

        # {'params': model.classifierB3.parameters(), 'lr': 0.0035},
        # {'params': model.classifierB4.parameters(), 'lr': 0.0035},
        # {'params': model.classifierB5.parameters(), 'lr': 0.0035},

        # {'params': model.classifierC2.parameters(), 'lr': 0.0035},
        # {'params': model.classifierC3.parameters(), 'lr': 0.0035},

        {'params': model.classifier.parameters(), 'lr': 0.0035},

        ])

    # optimizer = optim.SGD([
    #     {'params': base_params, 'lr': 0.1*opt.lr},

    #     {'params': model.classifierA0.parameters(), 'lr': opt.lr},
    #     {'params': model.classifierA1.parameters(), 'lr': opt.lr},
    #     {'params': model.classifierA2.parameters(), 'lr': opt.lr},
    #     {'params': model.classifierA3.parameters(), 'lr': opt.lr},

        # {'params': model.classifierB0.parameters(), 'lr': opt.lr},
        # {'params': model.classifierB1.parameters(), 'lr': opt.lr},
        # {'params': model.classifierB2.parameters(), 'lr': opt.lr},

        # {'params': model.classifierC0.parameters(), 'lr': opt.lr},
        # {'params': model.classifierC1.parameters(), 'lr': opt.lr},

        # {'params': model.classifierD0.parameters(), 'lr': opt.lr},

        # {'params': model.classifierB3.parameters(), 'lr': opt.lr},
        # {'params': model.classifierB4.parameters(), 'lr': opt.lr},
        # {'params': model.classifierB5.parameters(), 'lr': opt.lr},

        # {'params': model.classifierC2.parameters(), 'lr': opt.lr},
        # {'params': model.classifierC3.parameters(), 'lr': opt.lr},

        # {'params': model.classifier.parameters(), 'lr': opt.lr},

    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)


######################################################################
# Train and evaluate
# --------
#

dir_name = os.path.join('./logs', opt.name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile('./train_vrid.py', dir_name+'/train_vrid.py')
copyfile('models/base_model.py', dir_name+'/base_model.py')
copyfile('models/vrid_model.py', dir_name+'/vrid_model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()

triplet = TripletLoss(0.3)
xent = CrossEntropyLabelSmooth(num_classes=opt.nclasses)

def loss_func(score, feat, target):
    if opt.use_triplet_loss:
        if opt.label_smoothing:
            return xent(score, target) + triplet(feat, target)[0]
        else:
            return F.cross_entropy(score, target) + triplet(feat, target)[0]
    else:
        if opt.label_smoothing:
            return xent(score, target)
        else:
            return F.cross_entropy(score, target)

model = train_model(model, loss_func, optimizer,
                    exp_lr_scheduler, num_epochs=300)
