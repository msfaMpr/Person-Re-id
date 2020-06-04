import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from .base_model import ClassBlock


def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1
    return a


class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class PCB_Effi_GGNN(nn.Module):

    def __init__(self, model, freeze_backbone=False):
        super(PCB_Effi_GGNN, self).__init__()
        self.opt = model.opt
        self.model = model.model
        self.avgpool = model.avgpool
        self.dropout = model.dropout

        # self.graph = [[1, 1, 2], [2, 1, 3], [3, 1, 4], [4, 1, 5], [5, 1, 6],
        #               [6, 2, 5], [5, 2, 4], [4, 2, 3], [3, 2, 2], [2, 2, 1]]

        # self.graph = [[1, 1, 2], [2, 1, 3], [3, 1, 4],
        #               [4, 2, 3], [3, 2, 2], [2, 2, 1]]

        self.graph = [[1, 1, 2], [1, 2, 3], [1, 3, 4],
                      [2, 4, 3], [2, 5, 4], [3, 6, 4]]

        self.n_edge_types = 6

        self.state_dim = model.feature_dim
        self.n_node = self.opt.nparts
        self.n_steps = 5

        am = torch.Tensor(create_adjacency_matrix(
            self.graph, self.n_node, self.n_edge_types)).cuda()
        self.am = torch.unsqueeze(am, 0)

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(
            self.state_dim, self.n_node, self.n_edge_types)

        # # Output Model
        # self.out = nn.Sequential(
        #     nn.Linear(self.state_dim, self.state_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.state_dim, self.state_dim)
        # )

        self._initialization()

        self.classifier = ClassBlock(self.opt.nparts*self.state_dim, self.opt.nclasses, droprate=0.5,
                                     relu=False, bnorm=True, num_bottleneck=256)

        for i in range(self.opt.nparts):
            name = 'classifierA'+str(i)
            setattr(self, name, ClassBlock(self.state_dim, self.opt.nclasses,
                                           droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            # for i in range(self.opt.nparts-1):
            #     name = 'classifierB'+str(i)
            #     setattr(self, name, ClassBlock(2*1280, self.opt.nclasses, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            # for i in range(self.opt.nparts-1):
            #     name = 'classifierB'+str(i+self.opt.nparts-1)
            #     setattr(self, name, ClassBlock(2*1280, self.opt.nclasses, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            # for i in range(self.opt.nparts-2):
            #     name = 'classifierC'+str(i)
            #     setattr(self, name, ClassBlock(3*1280, self.opt.nclasses, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            # for i in range(self.opt.nparts-2):
            #     name = 'classifierC'+str(i+self.opt.nparts-2)
            #     setattr(self, name, ClassBlock(3*1280, self.opt.nclasses, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

            # for i in range(self.opt.nparts-3):
            #     name = 'classifierD'+str(i)
            #     setattr(self, name, ClassBlock(4*1280, self.opt.nclasses, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        if self.opt.freeze_backbone:
            with torch.no_grad():
                x = self.model.extract_features(x)
                x = self.avgpool(x)  # b*1280*4*1
                x = self.dropout(x)
                x = x.squeeze()  # b*1280*4
        else:
            x = self.model.extract_features(x)

        x = self.avgpool(x)  # b*1280*4*1
        x = self.dropout(x)
        x = x.squeeze()  # b*1280*4

        x = torch.transpose(x, 1, 2)  # b*4*1280

        # Gated Graph Neural Network
        gx = x
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](gx))
                out_states.append(self.out_fcs[i](gx))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node *
                                       self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node *
                                         self.n_edge_types, self.state_dim)

            am = self.am.repeat(gx.size(0), 1, 1)
            gx = self.propogator(in_states, out_states, gx, am)

        # gx = self.out(gx)
        gx = torch.transpose(gx, 1, 2)
        gx = torch.flatten(gx, 1)

        y = {}
        y['GGNN'] = self.classifier(gx)

        partA, partB, partC, partD = {}, {}, {}, {}
        predictA, predictB, predictC, predictD = {}, {}, {}, {}
        y['PCB'] = []
        # get six part feature batchsize*1280*4

        for i in range(self.opt.nparts):
            partA[i] = torch.flatten(x[:, i:i+1, :], 1)
            name = 'classifierA'+str(i)
            c = getattr(self, name)
            predictA[i] = c(partA[i])
            y['PCB'].append(predictA[i])

        # for i in range(self.opt.nparts-1):
        #     partB[i] = torch.flatten(x[:, i:i+2, :], 1)
        #     name = 'classifierB'+str(i)
        #     c = getattr(self, name)
        #     predictB[i] = c(partB[i])
        #     y['PCB'].append(predictB[i])

        # for i in range(self.opt.nparts-2):
        #     partC[i] = torch.flatten(x[:, i:i+3, :], 1)
        #     name = 'classifierC'+str(i)
        #     c = getattr(self, name)
        #     predictC[i] = c(partC[i])
        #     y['PCB'].append(predictC[i])

        # for i in range(self.opt.nparts-3):
        #     partD[i] = torch.flatten(x[:, i:i+4, :], 1)
        #     name = 'classifierD'+str(i)
        #     c = getattr(self, name)
        #     predictD[i] = c(partD[i])
        #     y['PCB'].append(predictD[i])

        # partB[3] = torch.flatten(torch.cat((x[:, :1, :], x[:, 2:3, :]), 1), 1)
        # predictB[3] = self.classifierB3(partB[3])
        # y['PCB'].append(predictB[3])

        # partB[4] = torch.flatten(torch.cat((x[:, :1, :], x[:, 3:4, :]), 1), 1)
        # predictB[4] = self.classifierB4(partB[4])
        # y['PCB'].append(predictB[4])

        # partB[5] = torch.flatten(torch.cat((x[:, 1:2, :], x[:, 3:4, :]), 1), 1)
        # predictB[5] = self.classifierB5(partB[5])
        # y['PCB'].append(predictB[5])

        # partC[2] = torch.flatten(torch.cat((x[:, :2, :], x[:, 3:4, :]), 1), 1)
        # predictC[2] = self.classifierC2(partC[2])
        # y['PCB'].append(predictC[2])

        # partC[3] = torch.flatten(torch.cat((x[:, :1, :], x[:, 2:, :]), 1), 1)
        # predictC[3] = self.classifierC3(partC[3])
        # y['PCB'].append(predictC[3])

        return y


class PCB_Effi_GGNN_test(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_GGNN_test, self).__init__()
        self.opt = model.opt
        self.model = model.model
        self.avgpool = model.avgpool

        '''
        self.graph = model.graph

        self.state_dim = model.state_dim
        self.n_edge_types = model.n_edge_types
        self.n_node = self.opt.nparts
        self.n_steps = model.n_steps
        # self.n_steps = 1

        self.am = model.am

        self.in_fcs = []
        self.out_fcs = []
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            self.in_fcs.append(model.in_fcs[i].cuda())
            self.out_fcs.append(model.out_fcs[i].cuda())

        # Propogation Model
        self.propogator = model.propogator

        # # Output Model
        # self.out = model.out

        '''

        for i in range(self.opt.nparts):
            name = 'classifierA'+str(i)
            c = getattr(model, name)
            setattr(self, name, c)

    def forward(self, x):

        x = self.model.extract_features(x)
        x = self.avgpool(x)
        x = x.squeeze()

        '''
        # Gated Graph Neural Network
        gx = torch.transpose(x, 1, 2)  # b*4*1280
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](gx))
                out_states.append(self.out_fcs[i](gx))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node *
                                       self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node *
                                         self.n_edge_types, self.state_dim)
            am = self.am.repeat(gx.size(0), 1, 1)
            gx = self.propogator(in_states, out_states, gx, am)
        gx = torch.transpose(gx, 1, 2)

        y = torch.cat([x, gx], 2)
        y = x
        '''

        part = {}
        predict = {}
        y = []
        
        x = x.transpose(1, 2)
        for i in range(self.opt.nparts):
            part[i] = torch.flatten(x[:, i:i+1, :], 1)
            name = 'classifierA'+str(i)
            c = getattr(self, name)
            predict[i] = c.add_block(part[i])
            y.append(predict[i])

        y = torch.cat(y, -1).view(-1, 256, 4)

        return y
