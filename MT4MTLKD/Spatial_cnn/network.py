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
                 args=None,
                 num_tool=6,
                 num_verb=10,
                 num_target=15,
                 num_triplet=100):
        super(VideoNas, self).__init__()

        # ResNet18
        self.args = args
        self.basemodel = BaseModel(self.args.network)
        self.num_f_mstct = self.args.teacher_dim
        self.feat_dim = self.args.student_dim
        if self.args.loss_type == 'all':
            self.wi = nn.Conv1d(self.feat_dim, self.num_f_mstct, kernel_size=1, stride=1, padding=0)
            self.wv = nn.Conv1d(self.feat_dim, self.num_f_mstct, kernel_size=1, stride=1, padding=0)
            self.wt = nn.Conv1d(self.feat_dim, self.num_f_mstct, kernel_size=1, stride=1, padding=0)
            self.mi = nn.Conv1d(self.num_f_mstct, self.feat_dim, kernel_size=1, stride=1, padding=0)
            self.mv = nn.Conv1d(self.num_f_mstct, self.feat_dim, kernel_size=1, stride=1, padding=0)
            self.mt = nn.Conv1d(self.num_f_mstct, self.feat_dim, kernel_size=1, stride=1, padding=0)
        if self.args.loss_type == 'i' or self.args.loss_type == 'all':
            self.classifier_i = Classifier(feat_dim=self.feat_dim, num_class=num_tool)
        if self.args.loss_type == 'v' or self.args.loss_type == 'all':
            self.classifier_v = Classifier(feat_dim=self.feat_dim, num_class=num_verb)
        if self.args.loss_type == 't' or self.args.loss_type == 'all':
            self.classifier_t = Classifier(feat_dim=self.feat_dim, num_class=num_target)
        if self.args.loss_type == 'ivt' or self.args.loss_type == 'all':
            self.classifier_ivt = Classifier(feat_dim=self.feat_dim, num_class=num_triplet)

    def forward(self, inputs, tool=None, verb=None, target=None):
        # (32, 3, 256, 448)
        high_feat = self.basemodel(inputs)
        out_feat = high_feat.squeeze(-1).squeeze(-1)
        if self.args.loss_type == 'all' and self.args.train:
            stus_fi_list = []
            stus_fv_list = []
            stus_ft_list = []
            f_list = [high_feat.squeeze(-1).squeeze(-1)] * 4
            tool_list = [tool]
            verb_list = [verb]
            target_list = [target]
            for s, t_i, t_v, t_t in zip(f_list, tool_list, verb_list, target_list):
                stus = torch.stack([s for i in range(self.feat_dim)]).permute(1, 2, 0)
                teas = torch.stack(
                    [self.mi(t_i.unsqueeze(-1)).squeeze(-1)] + [self.mv(t_v.unsqueeze(-1)).squeeze(-1)] + [
                        self.mt(t_t.unsqueeze(-1)).squeeze(-1)]).permute(1, 2, 0)

                attn = torch.einsum('bcd,bdn->bcn', stus / (stus.size(-1) ** 0.5), teas)
                attn = attn.softmax(dim=-1)
                s_new_i = self.wi((s * attn[:, :, 0]).unsqueeze(-1)).squeeze(-1)
                s_new_v = self.wv((s * attn[:, :, 1]).unsqueeze(-1)).squeeze(-1)
                s_new_t = self.wt((s * attn[:, :, 2]).unsqueeze(-1)).squeeze(-1)
                stus_fi_list.append(s_new_i)
                stus_fv_list.append(s_new_v)
                stus_ft_list.append(s_new_t)
            stus_fi = torch.concat(stus_fi_list, dim=1)
            stus_fv = torch.concat(stus_fv_list, dim=1)
            stus_ft = torch.concat(stus_ft_list, dim=1)
        else:
            stus_fi = 0
            stus_fv = 0
            stus_ft = 0
        # (32, 512, 8, 14)
        # feat_ivt = out_feat
        B = out_feat.shape[0]
        cls_ivt = torch.zeros((B, 100)).cuda()
        cls_i = torch.zeros((B, 6)).cuda()
        cls_v = torch.zeros((B, 10)).cuda()
        cls_t = torch.zeros((B, 15)).cuda()
        if self.args.loss_type == 'ivt' or self.args.loss_type == 'all':
            _, cls_ivt = self.classifier_ivt(high_feat)
        if self.args.loss_type == 'i' or self.args.loss_type == 'all':
            _, cls_i = self.classifier_i(high_feat)
        if self.args.loss_type == 'v' or self.args.loss_type == 'all':
            _, cls_v = self.classifier_v(high_feat)
        if self.args.loss_type == 't' or self.args.loss_type == 'all':
            _, cls_t = self.classifier_t(high_feat)

        return (stus_fi, cls_i), (stus_fv, cls_v), (stus_ft, cls_t), (out_feat, cls_ivt)


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
            self.basemodel.layer1[2].bn3.register_forward_hook(self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn3.register_forward_hook(self.get_activation('high_level_feature'))
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
