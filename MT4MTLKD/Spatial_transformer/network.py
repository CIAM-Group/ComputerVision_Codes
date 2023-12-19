# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Qeruy2Label(nn.Module):
    def __init__(self, args, backbone, transfomer, num_class):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.num_tool = 6
        self.num_verb = 10
        self.num_target = 15
        self.num_triplet = 100
        self.num_f_mstct = self.args.teacher_dim
        self.feat_dim = self.args.student_dim
        if self.args.loss_type == 'i' or self.args.loss_type == 'all':
            self.decoder_i = Decoder(backbone.num_channels, transfomer, self.num_tool)
        if self.args.loss_type == 'v' or self.args.loss_type == 'all':
            self.decoder_v = Decoder(backbone.num_channels, transfomer, self.num_verb)
        if self.args.loss_type == 't' or self.args.loss_type == 'all':
            self.decoder_t = Decoder(backbone.num_channels, transfomer, self.num_target)
        if self.args.loss_type == 'all':
            self.decoder_ivt = Decoder(backbone.num_channels, transfomer, self.num_triplet)

            self.wi = nn.Conv1d(self.feat_dim, self.num_f_mstct, kernel_size=1, stride=1, padding=0)
            self.wv = nn.Conv1d(self.feat_dim, self.num_f_mstct, kernel_size=1, stride=1, padding=0)
            self.wt = nn.Conv1d(self.feat_dim, self.num_f_mstct, kernel_size=1, stride=1, padding=0)
            self.mi = nn.Conv1d(self.num_f_mstct, self.feat_dim, kernel_size=1, stride=1, padding=0)
            self.mv = nn.Conv1d(self.num_f_mstct, self.feat_dim, kernel_size=1, stride=1, padding=0)
            self.mt = nn.Conv1d(self.num_f_mstct, self.feat_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, input, tool=None, verb=None, target=None):
        src, pos = self.backbone(input)
        B = input.shape[0]
        y_ivt = torch.zeros((B, 100)).cuda()
        y_i = torch.zeros((B, 6)).cuda()
        y_v = torch.zeros((B, 10)).cuda()
        y_t = torch.zeros((B, 15)).cuda()
        if self.args.loss_type == 'i' or self.args.loss_type == 'all':
            (feat_i, y_i) = self.decoder_i(src, pos)
            feat = feat_i
        if self.args.loss_type == 'v' or self.args.loss_type == 'all':
            (feat_v, y_v) = self.decoder_v(src, pos)
            feat = feat_v
        if self.args.loss_type == 't' or self.args.loss_type == 'all':
            (feat_t, y_t) = self.decoder_t(src, pos)
            feat = feat_t
        if self.args.loss_type == 'all':
            (feat_ivt, y_ivt) = self.decoder_ivt(src, pos)
            feat = feat_ivt

            stus_fi_list = []
            stus_fv_list = []
            stus_ft_list = []
            f_list = [feat]
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
            stus_fi, stus_fv, stus_ft = 0, 0, 0

        return (stus_fi, y_i), (stus_fv, y_v), (stus_ft, y_t), (feat, y_ivt)

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


class Decoder(nn.Module):
    def __init__(self, num_channels, transfomer, num_class):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.transformer = transfomer

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, src, pos):
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs, feat = self.transformer(self.input_proj(src), query_input, pos)  # B,K,d
        out = self.fc(hs[-1])
        feat = nn.AdaptiveAvgPool2d(1)(feat).squeeze(2).squeeze(2)
        return (feat, out)

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


def build_q2l(args):
    args.position_embedding = 'sine'
    args.pretrained = True

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Qeruy2Label(
        args=args,
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class
    )

    model.input_proj = nn.Identity()
    print("set model.input_proj to Indentify!")

    return model


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
