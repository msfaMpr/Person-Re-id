from __future__ import print_function, division

import time
import os
import argparse
import yaml
import math
import scipy.io
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

from datasets import init_dataset, ImageDataset, VideoDataset

from models.base_model import PCB, PCB_test, PCB_Effi, PCB_Effi_test
from models.vrid_model import VRidGGNN, VRidGGNN_test


######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--nparts', default=4, type=int, help='number of stipes')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--backbone', default='EfficientNet-B0', type=str, help='backbone model name')
parser.add_argument('--single_cls', action='store_true', help='use signle classifier')
parser.add_argument('--LSTM', action='store_true', help='use LSTM')
parser.add_argument('--GGNN', action='store_true', help='use GGNN')
parser.add_argument('--bidirectional', action='store_true', help='use bidirectional lstm')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--seq_len', default=4, type=int, help='number of frames in a sample')
parser.add_argument('--sample_method', default='evenly', type=str, help='method to sample frames')

opt = parser.parse_args()

###load config###
# load the training config
config_path = os.path.join('./logs', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

opt.LSTM = config['LSTM']
opt.GGNN = config['GGNN']
opt.nparts = config['nparts']
opt.single_cls = config['single_cls']
opt.backbone = config['backbone']
opt.freeze_backbone = config['freeze_backbone']
opt.use_triplet_loss = config['use_triplet_loss']
# opt.bidirectional = config['bidirectional']

if 'nclasses' in config:  # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


######################################################################
# Load Data
# ---------
#

data_transforms = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = init_dataset('mars', root='../')

queryloader = DataLoader(
    VideoDataset(dataset.query, seq_len=opt.seq_len, sample_method=opt.sample_method,
    transform=data_transforms), batch_size=opt.batchsize, shuffle=False, num_workers=8, drop_last=False)

galleryloader = DataLoader(
    VideoDataset(dataset.gallery, seq_len=opt.seq_len,sample_method=opt.sample_method,
    transform=data_transforms), batch_size=opt.batchsize, shuffle=False, num_workers=8, drop_last=False)

use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# --------
#

def load_network(network):
    save_path = os.path.join('./logs', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# --------
#

# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    # count = 0
    for data in tqdm(dataloaders):
        img, label, _ = data
        n, t, c, h, w = img.size()
        # count += n
        # print(count)s
        ff = torch.FloatTensor(n, 1280, 4).zero_().cuda()

        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(
                        input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


# gallery_path = image_datasets['gallery'].imgs
# query_path = image_datasets['query'].imgs

# gallery_cam, gallery_label = get_id(gallery_path)
# query_cam, query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
# --------
#

print('-------test-----------')

if opt.backbone == 'ResNet50':
    model_structure = PCB(opt)
elif opt.backbone == 'EfficientNet-B0':
    model_structure = PCB_Effi(opt)

model_structure = VRidGGNN(model_structure)
model = load_network(model_structure)
model = VRidGGNN_test(model)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, galleryloader)
    query_feature = extract_feature(model, queryloader)

# # Save to Matlab for check
# result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
#           'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
# scipy.io.savemat('pytorch_result.mat', result)

print(opt.name)
result = './logs/%s/result.txt' % opt.name
os.system('python evaluate_gpu.py | tee -a %s' % result)
# os.system('python evaluate_rerank.py | tee -a %s' % result)
