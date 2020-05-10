from __future__ import print_function, division

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from shutil import copyfile
import math
import yaml
from random_erasing import RandomErasing
from models.ggnn_model import PCB_Effi_GGNN
from models.lstm_model import PCB_Effi_LSTM
from models.base_model import PCB, PCB_Effi
import os
import time

#from PIL import Image

version = torch.__version__


######################################################################
# Options
# --------
#

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='output model name')
parser.add_argument('--data_dir', default='../Market/pytorch',
                    type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true',
                    help='use all training data')
parser.add_argument('--color_jitter', action='store_true',
                    help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--npart', default=4, type=int, help='number os stripes')
parser.add_argument('--erasing_p', default=0.0, type=float,
                    help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int,
                    help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--single_cls', action='store_true',
                    help='use single classifier')
parser.add_argument('--Backbone', default='EfficientNet-B0',
                    type=str, help='backbone model')
parser.add_argument('--LSTM', action='store_true', help='use LSTM')
parser.add_argument('--GGNN', action='store_true', help='use GGNN')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='train backbone network')
                 
opt = parser.parse_args()


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

transform_train_list = [
    transforms.Resize((384, 192), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_val_list = [
    transforms.Resize(size=(384, 192), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + \
        [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(
    os.path.join(opt.data_dir, 'train' + train_all),
    data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(
    os.path.join(opt.data_dir, 'val'),
    data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=opt.batchsize,
    shuffle=True, num_workers=8, pin_memory=True
)for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)


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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(
        dataset_sizes['train'] / opt.batchsize)*opt.warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
                inputs, labels = data
                now_batch_size, _, _, _ = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                # if fp16:
                #    inputs = inputs.half()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.single_cls:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 4

                    for i in range(num_part):
                        part[i] = outputs['PCB'][i]

                    score = sm(part[0]) + sm(part[1]) + \
                        sm(part[2]) + sm(part[3])

                    _, preds = torch.max(score.data, 1)

                    loss = criterion(outputs['LSTM'], labels)
                    for i in range(num_part):
                        loss += criterion(part[i], labels)

                    for i in range(num_part-1):
                        loss += criterion(outputs['PCB'][num_part+i], labels)

                    for i in range(num_part-2):
                        loss + criterion(outputs['PCB']
                                         [2*num_part+i-1], labels)

                    for i in range(num_part-3):
                        loss + criterion(outputs['PCB']
                                         [3*num_part+i-3], labels)

                    for i in range(5):
                        loss += criterion(outputs['PCB'][10+i], labels)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # for the new version like 0.4.0, 0.5.0 and 1.0.0
                if int(version[0]) > 0 or int(version[2]) > 3:
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
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
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
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

opt.nclasses = len(class_names)

if opt.Backbone == "ResNet50":
    model = PCB(opt)
elif opt.Backbone == "EfficientNet-B0":
    model = PCB_Effi(opt)

if opt.LSTM:
    # model_name = 'PCB-128_dim_cls'
    # model = load_network(model, model_name)
    model = PCB_Effi_LSTM(model, opt)
    # model_name = 'LSTM'
    # model = load_network(model, model_name)

if opt.GGNN:
    model_name = 'PCB-128_dim_cls'
    model = load_network(model, model_name)
    model = PCB_Effi_GGNN(model, opt)
    # model_name = 'LSTM'
    # model = load_network(model, model_name)

print(model)

if opt.single_cls:
    if opt.Backbone == 'ResNet50':
        ignored_params = list(map(id, model.model.fc.parameters()))
    else:
        ignored_params = list(map(id, model.model._fc.parameters()))
    ignored_params += (list(map(id, model.classifier.parameters())))
    if opt.freeze_backbone:
        ignored_params += (list(map(id, model.model.parameters())))
    base_params = filter(
        lambda p: id(p) not in ignored_params, model.parameters()
    )
    optimizer = optim.SGD(
        [{'params': base_params, 'lr': 0.1*opt.lr},
         {'params': model.classifier.parameters(), 'lr': opt.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    if opt.Backbone == 'ResNet50':
        ignored_params = list(map(id, model.model.fc.parameters()))
    else:
        ignored_params = list(map(id, model.model._fc.parameters()))
    ignored_params += (
        list(map(id, model.classifierA0.parameters()))
        + list(map(id, model.classifierA1.parameters()))
        + list(map(id, model.classifierA2.parameters()))
        + list(map(id, model.classifierA3.parameters()))

        + list(map(id, model.classifierB0.parameters()))
        + list(map(id, model.classifierB1.parameters()))
        + list(map(id, model.classifierB2.parameters()))

        + list(map(id, model.classifierC0.parameters()))
        + list(map(id, model.classifierC1.parameters()))

        + list(map(id, model.classifierD0.parameters()))

        + list(map(id, model.classifierB3.parameters()))
        + list(map(id, model.classifierB4.parameters()))
        + list(map(id, model.classifierB5.parameters()))

        + list(map(id, model.classifierC2.parameters()))
        + list(map(id, model.classifierC3.parameters()))

        + list(map(id, model.classifier.parameters()))
        )
    if opt.freeze_backbone:
        ignored_params += (list(map(id, model.model.parameters())))
    base_params = filter(
        lambda p: id(p) not in ignored_params, model.parameters()
    )
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1*opt.lr},

        {'params': model.classifierA0.parameters(), 'lr': opt.lr},
        {'params': model.classifierA1.parameters(), 'lr': opt.lr},
        {'params': model.classifierA2.parameters(), 'lr': opt.lr},
        {'params': model.classifierA3.parameters(), 'lr': opt.lr},

        {'params': model.classifierB0.parameters(), 'lr': opt.lr},
        {'params': model.classifierB1.parameters(), 'lr': opt.lr},
        {'params': model.classifierB2.parameters(), 'lr': opt.lr},

        {'params': model.classifierC0.parameters(), 'lr': opt.lr},
        {'params': model.classifierC1.parameters(), 'lr': opt.lr},

        {'params': model.classifierD0.parameters(), 'lr': opt.lr},

        {'params': model.classifierB3.parameters(), 'lr': opt.lr},
        {'params': model.classifierB4.parameters(), 'lr': opt.lr},
        {'params': model.classifierB5.parameters(), 'lr': opt.lr},

        {'params': model.classifierC2.parameters(), 'lr': opt.lr},
        {'params': model.classifierC3.parameters(), 'lr': opt.lr},

        {'params': model.classifier.parameters(), 'lr': opt.lr},

    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


######################################################################
# Train and evaluate
# --------
#

dir_name = os.path.join('./logs', opt.name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('models/base_model.py', dir_name+'/base_model.py')
if opt.LSTM:
    copyfile('models/lstm_model.py', dir_name+'/lstm_model.py')
if opt.GGNN:
    copyfile('models/ggnn_model.py', dir_name+'/ggnn_model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()

criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer,
                    exp_lr_scheduler, num_epochs=50)
