import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__package__)


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos),
                       N_anneal=100000, N_anneal_min_freq=0,
                       use_annealing=True):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.use_annealing = use_annealing

        self.N_anneal = N_anneal
        self.N_anneal_min_freq = N_anneal_min_freq

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, iteration):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        alpha = (len(self.freq_bands)-self.N_anneal_min_freq)*iteration/self.N_anneal
        for i in range(len(self.freq_bands)):
            w = (1-np.cos(np.pi*np.clip(alpha-i+self.N_anneal_min_freq, 0, 1)))/2.

            if not self.use_annealing:
                w = 1

            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq) * w)
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


# default tensorflow initialization of linear layers
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight.data)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=10)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class MyBatchNorm1d(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W
        self.bn = nn.BatchNorm1d(W)

    def forward(self, x):
        y = x.reshape(-1, x.shape[-1])
        y = self.bn(y)
        y = y.reshape(*x.shape)
        return y


class SineAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class MLPNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=False, use_shadow=False, act='relu'):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.use_shadow = use_shadow
        self.skips = skips
        if act == 'relu':
            actclass = nn.ReLU
        elif act == 'sine':
            actclass = SineAct
        elif act == 'elu':
            actclass = nn.ELU

        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            # if i % 4 == 0:
            #     self.base_layers.append(
            #         nn.Sequential(nn.Linear(dim, W), MyBatchNorm1d(W), nn.ReLU())
            #     )
            # else:
            #     self.base_layers.append(
            #         nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            #     )
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), actclass())
                # nn.Sequential(nn.Linear(dim, W), SineAct())
            )
            dim = W
            if i in self.skips and i != (D-1):      # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init

        sigma_layers = [nn.Linear(dim, 1), ]       # sigma must be positive
        sigma_layers.append(nn.ReLU())
        self.sigma_layers = nn.Sequential(*sigma_layers)
        # self.sigma_layers.apply(my_weights_init)      # xavier init

        # rgb color
        rgb_layers = []
        base_remap_layers = [nn.Linear(dim, 256), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(weights_init)

        dim = 256 + self.input_ch_viewdirs
        for i in range(1):
            rgb_layers.append(nn.Linear(dim, W // 2))
            # rgb_layers.append(MyBatchNorm1d(W // 2))
            rgb_layers.append(actclass())
            # rgb_layers.append(SineAct())
            dim = W // 2
        rgb_layers.append(nn.Linear(dim, 3))
        rgb_layers.append(nn.Sigmoid())     # rgb values are normalized to [0, 1]
        self.rgb_layers = nn.Sequential(*rgb_layers)
        # self.rgb_layers.apply(weights_init)

        shadow_layers = []
        dim = 256 + 9
        for i in range(1):
            shadow_layers.append(nn.Linear(dim, W // 2))
            # rgb_layers.append(MyBatchNorm1d(W // 2))
            shadow_layers.append(actclass())
            # rgb_layers.append(SineAct())
            dim = W // 2
        shadow_layers.append(nn.Linear(dim, 1))
        shadow_layers.append(nn.Sigmoid())     # shadow values are normalized to [0, 1]
        self.shadow_layers = nn.Sequential(*shadow_layers)

    def forward(self, input):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        input_pts = input[..., :self.input_ch]

        # print(input_pts.shape)
        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=-1)
            base = self.base_layers[i+1](base)

        sigma = self.sigma_layers(base)
        sigma = torch.abs(sigma)

        base_remap = self.base_remap_layers(base)
        input_viewdirs = input[..., -self.input_ch_viewdirs:]
        assert(not self.use_viewdirs)
        if not self.use_viewdirs:
            input_viewdirs = input_viewdirs * 0
        rgb = self.rgb_layers(torch.cat((base_remap, input_viewdirs), dim=-1))

        if self.use_shadow:
            input_sph = input[..., -self.input_ch_viewdirs-9:-self.input_ch_viewdirs]
            shadow = self.shadow_layers(torch.cat((base_remap, input_sph), dim=-1))
            shadow = shadow.repeat((1,)*(len(shadow.size())-1)+(3,))  # gray to rgb

            ret = OrderedDict([('rgb', rgb),
                               ('sigma', sigma.squeeze(-1)),
                               ('shadow', shadow)])
        else:
            ret = OrderedDict([('rgb', rgb),
                               ('sigma', sigma.squeeze(-1))])
        return ret
