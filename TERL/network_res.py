#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as basemodels
import torchvision.transforms as transforms


class VideoNas(nn.Module):
    def __init__(self,
                 num_tool=6,
                 num_verb=10,
                 num_target=15,
                 num_triplet=100,
                 basename='resnet18'):
        super(VideoNas, self).__init__()

        # ResNet18
        self.basemodel = BaseModel(basename)
        if basename == 'resnet18':
            feat_dim = 512
        else:
            feat_dim = 2048

        self.classifier_i = Classifier(feat_dim=feat_dim, num_class=num_tool)
        self.classifier_v = Classifier(feat_dim=feat_dim, num_class=num_verb)
        self.classifier_t = Classifier(feat_dim=feat_dim, num_class=num_target)

        self.classifier_ivt = Classifier(feat_dim=feat_dim, num_class=num_triplet)

    def forward(self, inputs):
        # (32, 3, 256, 448)
        high_feat = self.basemodel(inputs)
        # (32, 512, 8, 14)
        cls_ivt = self.classifier_ivt(high_feat)
        cls_i = self.classifier_i(high_feat)
        cls_v = self.classifier_v(high_feat)
        cls_t = self.classifier_t(high_feat)

        return cls_i, cls_v, cls_t, cls_ivt


class BaseModel(nn.Module):
    def __init__(self, basename='resnet18', *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {}
        if basename == 'resnet18':
            self.basemodel = basemodels.resnet18(pretrained=True)
            self.basemodel.layer1[1].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            self.basemodel.avgpool.register_forward_hook(self.get_activation('final_feature'))
        if basename == 'resnet50':
            self.basemodel = basemodels.resnet50(pretrained=True)
            self.basemodel.layer1[2].bn2.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(self.get_activation('high_level_feature'))
            self.basemodel.avgpool.register_forward_hook(self.get_activation('final_feature'))

    def get_activation(self, layer_name):
        def hook(module, input, output):
            self.output_feature[layer_name] = output

        return hook

    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['final_feature']


class Classifier(nn.Module):
    def __init__(self, feat_dim=512, num_class=100):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_class)

    def forward(self, inputs):
        x = torch.flatten(inputs, 1)
        y = self.fc(x)
        return x, y
