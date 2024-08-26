import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.backbone import build_backbone
from models.transformer import build_transformer
import numpy as np


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
    def __init__(self, input_channels, transfomer, num_class):
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
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.fc_mlp = nn.Linear(hidden_dim, 128)

    def forward(self, src, pos):
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight

        hs, feat, sim_mat_2 = self.transformer(self.input_proj(src), query_input, pos)  # B,K,d
        out = self.fc(hs[-1])

        feat = nn.AdaptiveAvgPool2d(1)(feat).squeeze(2).squeeze(2)
        out_mlp = self.fc_mlp(feat)
        # out_mlp = self.fc_mlp(hs[-1])
        x = feat

        y_ivt = out

        return (x, out_mlp, y_ivt, sim_mat_2)

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, num_class=1000, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.args = args

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_backbone(args)
        self.register_buffer("bank", torch.tensor(self.args.bank))
        if self.args.ht:

            self.cam_i_head = nn.Conv2d(self.encoder_q.num_channels, 6, kernel_size=1)
            self.cam_i_tail = nn.Conv2d(self.encoder_q.num_channels, 6, kernel_size=1)
            self.cam_v_head = nn.Conv2d(self.encoder_q.num_channels, 10, kernel_size=1)
            self.cam_v_tail = nn.Conv2d(self.encoder_q.num_channels, 10, kernel_size=1)
            self.cam_t_head = nn.Conv2d(self.encoder_q.num_channels, 15, kernel_size=1)
            self.cam_t_tail = nn.Conv2d(self.encoder_q.num_channels, 15, kernel_size=1)
            self.cam_ivt_head = nn.Conv2d(self.encoder_q.num_channels, 100, kernel_size=1)
            self.cam_ivt_tail = nn.Conv2d(self.encoder_q.num_channels, 100, kernel_size=1)
        else:
            self.cam_i = nn.Conv2d(self.encoder_q.num_channels, 6, kernel_size=1)
            self.cam_v = nn.Conv2d(self.encoder_q.num_channels, 10, kernel_size=1)
            self.cam_t = nn.Conv2d(self.encoder_q.num_channels, 15, kernel_size=1)
            self.cam_ivt = nn.Conv2d(self.encoder_q.num_channels, 100, kernel_size=1)
            self.cam_disen = nn.Conv2d(self.encoder_q.num_channels + 1, self.encoder_q.num_channels, kernel_size=1)

        # self.cam_num = nn.Conv2d(self.encoder_q.num_channels, 4, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        if mlp:  # hack: brute-force replacement
            self.encoder_k = build_backbone(args)

            self.EMA(self.encoder_q, self.encoder_k)
            self.cam_disen_k = nn.Conv2d(self.encoder_q.num_channels + 1, self.encoder_q.num_channels, kernel_size=1)
            self.EMA(self.cam_disen, self.cam_disen_k)

            # create the queue
            self.register_buffer("queue", torch.randn(dim, K))

            self.register_buffer("i_prototpye", torch.rand(6, self.args.moco_dim))
            self.register_buffer("v_prototpye", torch.rand(10, self.args.moco_dim))
            self.register_buffer("t_prototpye", torch.rand(15, self.args.moco_dim))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_l", torch.zeros(1, K).long())
            self.register_buffer("queue_l_i", torch.zeros(1, K).long())
            self.register_buffer("queue_l_v", torch.zeros(1, K).long())
            self.register_buffer("queue_l_t", torch.zeros(1, K).long())
            # self.register_buffer("queue_t", torch.zeros(self.args.moco_class, K).long())
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def EMA(self, net1, net2):
        for param_q, param_k in zip(net1.parameters(), net2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # for param_q, param_k in zip(self.dec_q_tail_i.parameters(), self.dec_k_tail_i.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # for param_q, param_k in zip(self.dec_q_tail_v.parameters(), self.dec_k_tail_v.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # for param_q, param_k in zip(self.dec_q_tail_t.parameters(), self.dec_k_tail_t.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # for param_q, param_k in zip(self.dec_q_head.parameters(), self.dec_k_head.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # for param_q, param_k in zip(self.q_head_hack.parameters(), self.k_head_hack.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, labels_i, labels_v, labels_t):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        # keys_i = concat_all_gather(keys[0])
        # keys_v = concat_all_gather(keys[1])
        # keys_t = concat_all_gather(keys[2])
        # keys_t = concat_all_gather(keys_t)
        labels = concat_all_gather(labels)
        labels_i = concat_all_gather(labels_i)
        labels_v = concat_all_gather(labels_v)
        labels_t = concat_all_gather(labels_t)
        # true_labels = concat_all_gather(true_labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size, :]
            # keys_i = keys_i[:batch_size, :]
            # keys_v = keys_v[:batch_size, :]
            # keys_t = keys_t[:batch_size, :]
            # keys_t = concat_all_gather(keys_t)
            labels = labels[:batch_size, :]
            labels_i = labels_i[:batch_size, :]
            labels_v = labels_v[:batch_size, :]
            labels_t = labels_t[:batch_size, :]
        # assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        # self.queue_head[:, ptr:ptr + batch_size, :] = keys.permute(2, 0, 1)
        # self.queue_tail[:, ptr:ptr + batch_size, :] = keys_t.permute(2, 0, 1)

        self.queue[:, ptr:ptr + batch_size] = keys.T
        # self.queue_i[:, ptr:ptr + batch_size] = keys_i.T
        # self.queue_v[:, ptr:ptr + batch_size] = keys_v.T
        # self.queue_t[:, ptr:ptr + batch_size] = keys_t.T
        self.queue_l[:, ptr:ptr + batch_size] = labels.T
        self.queue_l_i[:, ptr:ptr + batch_size] = labels_i.T
        self.queue_l_v[:, ptr:ptr + batch_size] = labels_v.T
        self.queue_l_t[:, ptr:ptr + batch_size] = labels_t.T
        # self.queue_t[:, ptr:ptr + batch_size] = true_labels.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        gpu_idx = 0
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = 0
        # gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def headtail(self, src, cam_funs, masks):
        cam_i_head = cam_funs[0](src[0])
        y_i_head = self.pool(cam_i_head).view(cam_i_head.shape[0], -1)
        cam_i_tail = cam_funs[1](src[0])
        y_i_tail = self.pool(cam_i_tail).view(cam_i_tail.shape[0], -1)
        head_mask = masks[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        tail_mask = masks[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cam_i = cam_i_head * head_mask + cam_i_tail * tail_mask
        y_i = y_i_head * torch.ones_like(y_i_head) * masks[0] + y_i_tail * torch.ones_like(
            y_i_tail) * masks[1]
        return cam_i, y_i

    def valid_q(self, cam, labels, encoder, src):
        idxes_ivt = torch.where(labels[0] == 1)
        idxes_i = (idxes_ivt[0], self.bank[idxes_ivt[-1], 1])
        mlp_feats = torch.stack([
            encoder(torch.concatenate([src[0], cam[:, i, :, :].unsqueeze(1)], dim=1)) for i in
            range(cam.shape[1])]).permute(1, 0, 2, 3, 4)[idxes_i]

        lab_ivt = idxes_ivt[-1]
        return mlp_feats, lab_ivt

    def valid_q_ori(self, cam, labels, encoder, src):
        w_cam = cam.view(cam.shape[0], cam.shape[1], -1)
        w_min = w_cam.min()
        w_max = w_cam.max()
        w_cam = (w_cam - w_min) / (w_max - w_min)
        w_cam = w_cam.view(cam.shape)
        idxes = torch.where(labels[0] == 1)
        mlp_feats = torch.stack([
            encoder[0].avgpool(src[0] * w_cam[:, i, :, :].unsqueeze(1)).squeeze(-1).squeeze(-1) for i in
            range(w_cam.shape[1])]).permute(1, 0, 2)[idxes]
        # mlp_feats = torch.stack([self.encoder_q[0].head(
        #     encoder[0].avgpool(src[0] * w_cam[:, i, :, :].unsqueeze(1)).squeeze(-1).squeeze(-1)) for i in
        #     range(w_cam.shape[1])]).permute(1, 0, 2)[idxes]
        lab_ivt = idxes[-1]
        return mlp_feats, lab_ivt

    def forward(self, im_q, im_k=None, labels=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute key features
        mlp_feat, src, pos = self.encoder_q(im_q)

        # cam_num = self.cam_num(src[0])
        # y_num = self.pool(cam_num).view(cam_num.shape[0], -1)
        if self.args.ht:
            cam_i, y_i = self.headtail(src, [self.cam_i_head, self.cam_i_tail],
                                       [self.args.head_mask_i, self.args.tail_mask_i])

            cam_v, y_v = self.headtail(src, [self.cam_v_head, self.cam_v_tail],
                                       [self.args.head_mask_v, self.args.tail_mask_v])

            cam_t, y_t = self.headtail(src, [self.cam_t_head, self.cam_t_tail],
                                       [self.args.head_mask_t, self.args.tail_mask_t])

            cam, y = self.headtail(src, [self.cam_ivt_head, self.cam_ivt_tail],
                                   [self.args.head_mask, self.args.tail_mask])
        else:
            cam_i = self.cam_i(src[0])
            y_i = self.pool(cam_i).view(cam_i.shape[0], -1)
            cam_v = self.cam_v(src[0])
            y_v = self.pool(cam_v).view(cam_v.shape[0], -1)
            cam_t = self.cam_t(src[0])
            y_t = self.pool(cam_t).view(cam_t.shape[0], -1)
            cam = self.cam_ivt(src[0])
            y = self.pool(cam).view(cam.shape[0], -1)

        feat = self.pool(src[0]).squeeze(2).squeeze(2)

        if im_k != None:
            self.i_prototpye = torch.vstack(
                [self.queue[:, torch.where(self.queue_l_i == i)[1]].mean(axis=1) if len(
                    torch.where(self.queue_l_i == i)[1]) > 0 else self.i_prototpye[i, :] for i in
                 range(6)])
            self.v_prototpye = torch.vstack(
                [self.queue[:, torch.where(self.queue_l_v == i)[1]].mean(axis=1) if len(
                    torch.where(self.queue_l_v == i)[1]) > 0 else self.v_prototpye[i, :] for i in
                 range(10)])
            self.t_prototpye = torch.vstack(
                [self.queue[:, torch.where(self.queue_l_t == i)[1]].mean(axis=1) if len(
                    torch.where(self.queue_l_t == i)[1]) > 0 else self.t_prototpye[i, :] for i in
                 range(15)])
            mlp_feats, lab_ivt = self.valid_q(cam, labels, self.cam_disen, src)
            y_tail = self.pool(self.cam_ivt(mlp_feats)).view(mlp_feats.shape[0], -1)
            mlp_feats = self.pool(mlp_feats).squeeze(2).squeeze(2)
            q = nn.functional.normalize(mlp_feats, dim=-1)
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                mlp_feat_k, src_k, pos_k = self.encoder_k(im_k)
                cam_k = self.cam_ivt(src[0])
                # cam_k, _ = self.headtail(src_k, [self.cam_ivt_head, self.cam_ivt_tail],
                #                          [self.args.head_mask, self.args.tail_mask])
                cam_k = self._batch_unshuffle_ddp(cam_k, idx_unshuffle)
                mlp_feats_k, lab_ivt_k = self.valid_q(cam_k, labels, self.cam_disen_k, src_k)
                mlp_feats_k = self.pool(mlp_feats_k).squeeze(2).squeeze(2)

                k = nn.functional.normalize(mlp_feats_k, dim=-1)

            # compute logits
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,cm->nm', [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=-1)

            queue_label = self.queue_l.clone().detach()
            queue_label_i = self.queue_l_i.clone().detach()
            queue_label_v = self.queue_l_v.clone().detach()
            queue_label_t = self.queue_l_t.clone().detach()
            lab_i = self.bank[lab_ivt, 1].unsqueeze(-1)
            lab_v = self.bank[lab_ivt, 2].unsqueeze(-1)
            lab_t = self.bank[lab_ivt, 3].unsqueeze(-1)

            self._dequeue_and_enqueue(k, lab_ivt.unsqueeze(-1), lab_i, lab_v, lab_t)
            # l_proto_ivt1 = torch.einsum('nkc,cj->nkj', [out_feat_ivt, self.ivt_prototpye.T])
            l_proto_i1 = torch.einsum('nc,cm->nm', [torch.vstack([mlp_feats, mlp_feats_k]), self.i_prototpye.T])
            l_proto_v1 = torch.einsum('nc,cm->nm', [torch.vstack([mlp_feats, mlp_feats_k]), self.v_prototpye.T])
            l_proto_t1 = torch.einsum('nc,cm->nm', [torch.vstack([mlp_feats, mlp_feats_k]), self.t_prototpye.T])

            return [logits, lab_ivt.unsqueeze(-1), y_tail], \
                   [[l_proto_i1, l_proto_v1, l_proto_t1],
                    [torch.vstack([lab_i, lab_i]), torch.vstack([lab_v, lab_v]), torch.vstack([lab_t, lab_t])]], \
                   [queue_label, queue_label_i, queue_label_v, queue_label_t], \
                   [(feat, y), (feat, y_i), (feat, y_v), (feat, y_t)]
        else:
            return 0, 0, 0, [(feat, y), (feat, y_i), (feat, y_v), (feat, y_t)]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #                   for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #
    # output = torch.cat(tensors_gather, dim=0)
    output = tensor
    return output
