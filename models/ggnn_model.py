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

    def __init__(self, model):
        super(PCB_Effi_GGNN, self).__init__()

        self.part = model.part  # We cut the pool5 to 6 parts
        self.model = model.model
        self.avgpool = model.avgpool
        self.dropout = model.dropout

        # self.graph = [[1, 1, 2], [2, 1, 3], [3, 1, 4], [4, 1, 5], [5, 1, 6],
        #               [6, 2, 5], [5, 2, 4], [4, 2, 3], [3, 2, 2], [2, 2, 1]]

        self.graph = [[1, 1, 2], [2, 1, 3], [3, 1, 4],
                      [4, 2, 3], [3, 2, 2], [2, 2, 1]]

        self.state_dim = 1280
        self.n_edge_types = 2
        self.n_node = self.part
        self.n_steps = 1

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

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim)
        )

        self._initialization()

        # # define 4 classifiers
        # for i in range(self.part):
        #     name = 'classifier'+str(i)
        #     c = getattr(model, name)
        #     setattr(self, name, c)

        self.classifier = ClassBlock(self.part*1280, model.class_num, droprate=0.5,
                                        relu=False, bnorm=True, num_bottleneck=256)

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.extract_features(x)
            x = self.avgpool(x)  # b*1280*4*1
            x = self.dropout(x)
            x = x.squeeze()  # b*1280*4

        # Gated Graph Neural Network
        x = torch.transpose(x, 1, 2)  # b*4*1280
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](x))
                out_states.append(self.out_fcs[i](x))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node *
                                       self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node *
                                         self.n_edge_types, self.state_dim)

            am = self.am.repeat(x.size(0), 1, 1)
            x = self.propogator(in_states, out_states, x, am)

        x = self.out(x)

        # x = torch.transpose(x, 1, 2)
        # part = {}
        # predict = {}
        # # get six part feature batchsize*1280*6
        # for i in range(self.part):
        #     part[i] = torch.squeeze(x[:, :, i])
        #     name = 'classifier'+str(i)
        #     c = getattr(self, name)
        #     predict[i] = c(part[i])
        # y = []
        # for i in range(self.part):
        #     y.append(predict[i])

        x = x.view((-1, self.part*1280))
        y = self.classifier(x)

        return y


class PCB_Effi_GGNN_test(nn.Module):
    def __init__(self, model):
        super(PCB_Effi_GGNN_test, self).__init__()
        self.part = model.part
        self.model = model.model
        self.avgpool = model.avgpool

        self.graph = model.graph

        self.state_dim = model.state_dim
        self.n_edge_types = model.n_edge_types
        self.n_node = self.part
        self.n_steps = model.n_steps

        self.am = model.am

        self.in_fcs = []
        self.out_fcs = []
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            self.in_fcs.append(model.in_fcs[i].cuda())
            self.out_fcs.append(model.out_fcs[i].cuda())

        # Propogation Model
        self.propogator = model.propogator

        # Output Model
        self.out = model.out

    def forward(self, x):

        x = self.model.extract_features(x)
        x = self.avgpool(x)
        x = x.squeeze()

        # Gated Graph Neural Network
        x = torch.transpose(x, 1, 2)  # b*4*1280
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](x))
                out_states.append(self.out_fcs[i](x))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node *
                                       self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node *
                                         self.n_edge_types, self.state_dim)

            am = self.am.repeat(x.size(0), 1, 1)
            x = self.propogator(in_states, out_states, x, am)

        x = self.out(x)
        x = torch.transpose(x, 1, 2)

        y = x.view(x.size(0), x.size(1), x.size(2))
        return y
