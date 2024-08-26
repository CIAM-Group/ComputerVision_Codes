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
from models import moco
from utils.misc import clean_state_dict


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
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
    def __init__(self, backbone, transfomer, num_class):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_tool = 6,
        self.num_verb = 10,
        self.num_target = 15,
        self.num_triplet = 100,

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        # hs = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d
        # out = self.fc(hs[-1])
        # x = hs[-1]
        hs, feat = self.transformer(self.input_proj(src), query_input, pos)  # B,K,d
        out = self.fc(hs[-1])
        feat = nn.AdaptiveAvgPool2d(1)(feat).squeeze(2).squeeze(2)
        # import ipdb; ipdb.set_trace()
        x = feat

        if out.shape[-1] == 6:
            y_i = out
            y_v = torch.stack([out[:, 0]] * 10).permute(1, 0)
            y_t = torch.stack([out[:, 0]] * 15).permute(1, 0)
            y_ivt = torch.stack([out[:, 0]] * 100).permute(1, 0)
        elif out.shape[-1] == 10:
            y_v = out
            y_i = torch.stack([out[:, 0]] * 6).permute(1, 0)
            y_t = torch.stack([out[:, 0]] * 15).permute(1, 0)
            y_ivt = torch.stack([out[:, 0]] * 100).permute(1, 0)
        elif out.shape[-1] == 15:
            y_t = out
            y_i = torch.stack([out[:, 0]] * 6).permute(1, 0)
            y_v = torch.stack([out[:, 0]] * 10).permute(1, 0)
            y_ivt = torch.stack([out[:, 0]] * 100).permute(1, 0)
        else:
            y_ivt = out
            y_i = torch.stack([out[:, 0]] * 6).permute(1, 0)
            y_v = torch.stack([out[:, 0]] * 10).permute(1, 0)
            y_t = torch.stack([out[:, 0]] * 15).permute(1, 0)

        return (x, y_i), (x, y_v), (x, y_t), (x, y_ivt)

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
    # args.img_size = 384
    # args.backbone = 'swin_L_384_22k'
    # args.hidden_dim = 1536
    args.pretrained = True

    # backbone = build_backbone(args)
    # transformer = build_transformer(args)
    # transformer.num_class = 100
    # transformer_i = build_transformer(args)
    # transformer_v = build_transformer(args)
    # transformer_t = build_transformer(args)
    # transformers = [transformer]

    # base_model = Qeruy2Label(
    #     backbone=backbone,
    #     transfomer=transformer,
    #     num_class=128
    # )
    model = moco.MoCo(args, 101, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

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
