#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn

import torch.nn.functional as F
import copy
import random
import math
import numpy as np

class VideoNas(nn.Module):

    def __init__(self, args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, num_i=6, num_v=10,
                 num_t=15):
        super(VideoNas, self).__init__()
        self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_out_i = nn.Conv1d(num_f_maps, num_i, 1)
        self.conv_out_v = nn.Conv1d(num_f_maps, num_v, 1)
        self.conv_out_t = nn.Conv1d(num_f_maps, num_t, 1)
        self.args = args
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(args, num_layers_R, num_f_maps, num_classes, num_classes, self.conv_out)) for s in
             range(num_R)])
        self.use_fpn = args.fpn
        self.use_output = args.output
        self.use_feature = args.feature
        self.use_trans = args.trans
        if args.fpn:
            self.fpn = FPN(num_f_maps)

    def forward(self, x, ismask):
        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        f_list = []
        x = x.permute(0, 2, 1)
        if self.args.mask and ismask:
            num_patches = x.flatten().shape[0]
            num_mask = int(num_patches * 0.75)
            mask = torch.concat((torch.zeros(num_patches - num_mask), torch.ones(num_mask)))
            mask = mask[torch.randperm(mask.nelement())]
            mask = mask.view(x.shape).cuda()

            f, out1 = self.PG(x, mask)
        else:
            f, out1 = self.PG(x)

        f_list.append(f)
        if not self.use_fpn:
            out_list.append(out1)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)
        if self.use_fpn:
            f_list = self.fpn(f_list)
            for f in f_list:
                out_list.append(self.conv_out(f))
                out_list_i.append(self.conv_out_i(f))
                out_list_v.append(self.conv_out_v(f))
                out_list_t.append(self.conv_out_t(f))
        return out_list, out_list_i, out_list_v, out_list_t, f_list, f_list


class FPN(nn.Module):
    def __init__(self, num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, W = y.size()
        return F.interpolate(x, size=W, mode='linear') + y

    def forward(self, out_list):
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1, p2, p3, p4]


class BaseCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        print(num_layers)
        super(BaseCausalTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.channel_dropout = nn.Dropout2d()
        self.num_classes = num_classes

    def forward(self, x, mask=None):

        if mask is not None:
            x = x * mask

        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)

        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)

        x = self.conv_out(out)  # (bs, c, l)

        return out, x


class Refinement(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes, conv_out):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.max_pool_1x1 = nn.AvgPool1d(kernel_size=7, stride=3)
        self.use_output = args.output
        self.hier = args.hier

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x
        for layer in self.layers:
            out = layer(out)
        if self.hier:
            f = self.max_pool_1x1(out)
        else:
            f = out
        out = self.conv_out(f)

        return f, out


class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, padding=None):
        super(DilatedResidualCausalLayer, self).__init__()
        if padding == None:

            self.padding = 2 * dilation
        else:
            self.padding = padding
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)  #
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0) # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return x + out
