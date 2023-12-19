#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch import nn
from MSTCT.Classification_Module import Classification_Module
from MSTCT.TS_Mixer import Temporal_Mixer
from MSTCT.Temporal_Encoder import TemporalEncoder
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def draw_tsne(inputs_2ds, figname):
    fig, ax = plt.subplots()
    for i, inputs_2d in enumerate(inputs_2ds):
        ax.scatter(
            inputs_2d[:, 0],
            inputs_2d[:, 1],
            s=3,
            c=info['colors'][i],
            edgecolors='none',
            label=info['labels'][i],
            alpha=1,
            rasterized=False
        )
    ax.legend(fontsize=10, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(figname, bbox_inches='tight')
    plt.close()


tsne = TSNE(
    n_components=2, metric='euclidean', verbose=1,
    perplexity=50, n_iter=1000, learning_rate=200.
)

info = {
    'colors': ['C1', 'C2', 'C3', 'C4'],
    'labels': ['I', 'V', 'T', 'IVT']
}


class VideoNas(nn.Module):
    """
    MS-TCT for action detection
    """

    def __init__(self, args, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_tool=6,
                 num_verb=10,
                 num_target=15,
                 num_triplet=100):
        super(VideoNas, self).__init__()

        self.args = args
        self.dropout = nn.Dropout()

        self.TemporalEncoder = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                               num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                               num_block=num_block)

        self.Temporal_Mixer = Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        if self.args.loss_type == 'i':
            self.classifier_i = Classifier(final_embedding_dim, num_tool)
        if self.args.loss_type == 'v':
            self.classifier_v = Classifier(final_embedding_dim, num_verb)
        if self.args.loss_type == 't':
            self.classifier_t = Classifier(final_embedding_dim, num_target)
        if self.args.loss_type == 'ivt':
            self.classifier_ivt = Classifier(final_embedding_dim, num_triplet)

    def forward(self, inputs):
        inputs = self.dropout(inputs)

        # Temporal Encoder Module
        x = self.TemporalEncoder(inputs)

        # Temporal Scale Mixer Module
        concat_feature = self.Temporal_Mixer(x)
        B = inputs.shape[0]
        T = inputs.shape[-1]
        y_ivt = torch.zeros((B, T, 100)).cuda()
        y_i = torch.zeros((B, T, 6)).cuda()
        y_v = torch.zeros((B, T, 10)).cuda()
        y_t = torch.zeros((B, T, 15)).cuda()
        feat_i = feat_v = feat_t = feat_ivt = concat_feature

        # Classification Module
        if self.args.loss_type == 'i':
            y_i, feat_i = self.classifier_i(concat_feature)
        if self.args.loss_type == 'v':
            y_v, feat_v = self.classifier_v(concat_feature)
        if self.args.loss_type == 't':
            y_t, feat_t = self.classifier_t(concat_feature)
        if self.args.loss_type == 'ivt':
            y_ivt, feat_ivt = self.classifier_ivt(concat_feature)

        return (y_i, feat_i), (y_v, feat_v), (y_t, feat_t), (y_ivt, concat_feature)


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.linear_fuse = nn.Conv1d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1)
        self.linear_pred = nn.Conv1d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, concat_feature):
        # Classification Branch
        x = self.linear_fuse(concat_feature)
        feat = self.dropout(x)
        x = self.linear_pred(feat)
        x = x.permute(0, 2, 1)

        return x, feat
