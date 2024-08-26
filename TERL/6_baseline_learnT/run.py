#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import libraries
import os
import sys
import time
import torch
import random
import network
import argparse
import platform
import ivtmetrics  # You must "pip install ivtmetrics" to use
import dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import models
from utils.misc import clean_state_dict
from imbsam import SAM, ImbSAM, SGD

np.seterr(invalid='ignore')
# %% @args parsing
# %% @args parsing
parser = argparse.ArgumentParser()
# model
parser.add_argument('--model', type=str, default='rendezvous', choices=['rendezvous'], help='Model name?')
parser.add_argument('--version', type=str, default='', help='Model version control (for keeping several versions)')

# job
parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test', action='store_true', help='to test')
parser.add_argument('--fix_backbone', action='store_true', help='to test')
parser.add_argument('--ht', action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,
                    help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/public/home/guisc/Data/Video/Surgical/CholecT45',
                    help='path to dataset?')
parser.add_argument('--rho', type=float, default=0.05)
parser.add_argument('--w_con', type=float, default=1.0)
parser.add_argument('--w_proto', type=float, default=1.0)
parser.add_argument('--w_tail', type=float, default=1.0)

parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval',
                    choices=['cholect50', 'cholect45', 'cholect50-challenge', 'cholect50-crossval',
                             'cholect45-crossval', 'cholect45-challenge'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1,
                    help='The test split in k-fold cross-validation')
parser.add_argument('--image_width', type=int, default=448, help='Image width ')
parser.add_argument('--image_height', type=int, default=256, help='Image height ')
parser.add_argument('--image_channel', type=int, default=3, help='Image channels ')
parser.add_argument('--num_tool_classes', type=int, default=6, help='Number of tool categories')
parser.add_argument('--num_verb_classes', type=int, default=10, help='Number of verb categories')
parser.add_argument('--num_target_classes', type=int, default=15, help='Number of target categories')
parser.add_argument('--num_triplet_classes', type=int, default=100, help='Number of triplet categories')
parser.add_argument('--augmentation_list', type=str, nargs='*',
                    default=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                    # default=['original', 'vflip', 'hflip', 'contrast1', 'rot90', 'brightness', 'contrast'],
                    help='List augumentation styles (see dataloader.py for list of supported styles).')
# hp
parser.add_argument('-b', '--batch', type=int, default=32, help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=100, help='How many training epochs?')
parser.add_argument('--w_epoch', type=int, default=5, help='How many training epochs?')
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[9, 18, 58],
                    help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('--drop_classes', type=int, nargs='+', default=[],
                    help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('--tail_classes_ivt', type=int, nargs='+', default=[],
                    help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01, 0.01, 0.01],
                    help='List learning rates for tool, verb-target, triplet respectively')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization weight decay constant')
parser.add_argument('--decay_steps', type=int, default=10, help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99, help='Learning rates weight decay rate')
parser.add_argument('--momentum', type=float, default=0.95, help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1, help='Learning rates weight decay power')
parser.add_argument('--drop_rate', type=float, default=0.0, help='Learning rates weight decay power')
# weights
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--loss_type', type=str, default='all', help='path to pretrain_weight?')
parser.add_argument('--opt_type', type=str, default='sgd', help='path to pretrain_weight?')
parser.add_argument('--backbone', type=str, default='all', help='path to pretrain_weight?')
parser.add_argument('--pos_w', type=str, default='ones', help='path to pretrain_weight?')
parser.add_argument('--img_size', type=int, default=384, help='path to pretrain_weight?')
parser.add_argument('--train_div', type=int, default=1, help='path to pretrain_weight?')
parser.add_argument('--tail_num', type=int, default=84, help='path to pretrain_weight?')
parser.add_argument('--hidden_dim', type=int, default=1536, help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')

# moco specific configs:
parser.add_argument('--moco_class', default=100, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', default='True',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default='True',
                    help='use cosine lr schedule')

parser.add_argument('--fc', action='store_true', help='to test')
# device
parser.add_argument('--gpu', type=str, default="0,1,2",
                    help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.local_rank = -1


def assign_gpu(gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


# %% @params definitions
# seed
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)
FLAGS.bank_id = {6: 1, 10: 2, 15: 3, 100: 0}
is_train = FLAGS.train
is_test = FLAGS.test
dataset_variant = FLAGS.dataset_variant
data_dir = FLAGS.data_dir
kfold = FLAGS.kfold if "crossval" in dataset_variant else 0
version = FLAGS.version
batch_size = FLAGS.batch
pretrain_dir = FLAGS.pretrain_dir
test_ckpt = FLAGS.test_ckpt
weight_decay = FLAGS.weight_decay
learning_rates = FLAGS.initial_learning_rates
warmups = FLAGS.warmups
decay_steps = FLAGS.decay_steps
decay_rate = FLAGS.decay_rate
power = FLAGS.power
momentum = FLAGS.momentum
epochs = FLAGS.epochs
gpu = FLAGS.gpu
image_height = FLAGS.image_height
image_width = FLAGS.image_width
image_channel = FLAGS.image_channel
num_triplet = FLAGS.num_triplet_classes
num_tool = FLAGS.num_tool_classes
num_verb = FLAGS.num_verb_classes
num_target = FLAGS.num_target_classes
val_interval = FLAGS.epochs - 1 if FLAGS.val_interval == -1 else FLAGS.val_interval
set_chlg_eval = True  # To observe challenge evaluation protocol
gpu = ",".join(str(FLAGS.gpu).split(","))

# %% assign device and set debugger options
assign_gpu(gpu=gpu)
np.seterr(divide='ignore', invalid='ignore')
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

FLAGS.multigpu = len(gpu) > 1  # not yet implemented !
mheaders = ["", "l", "cholect", "k"]
margs = [FLAGS.model, dataset_variant, kfold]
modelname = "_".join(["{}{}".format(x, y) for x, y in zip(mheaders, margs) if len(str(y))])
model_dir = "./__checkpoint__/run_{}".format(version)
if not os.path.exists(model_dir): os.makedirs(model_dir)
resume_ckpt = None
ckpt_path = os.path.join(model_dir, '{}.pth'.format(modelname))
ckpt_path_epoch = os.path.join(model_dir, '{}'.format(modelname))
logfile = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations = FLAGS.augmentation_list
iterable_augmentations = []
print("Configuring network ...")
with open('ins_num.txt', 'r') as f:
    ins_nums = [l.strip().split('\t') for l in f.readlines()]
    ins_num = [int(l[0]) for l in ins_nums]
    ins_num1 = [int(l[1]) for l in ins_nums]

    # ins_num1 = [int(l.strip().split('\t')[1]) for l in f.readlines()]
ins_i = ins_num[:6]
ins_i_num = torch.tensor(sorted(list(np.array([ins_i, ins_num1[:6]]).T), key=(lambda x: x[0])))[:, 1].cuda()
ins_v = ins_num[6:6 + 10]
ins_v_num = torch.tensor(sorted(list(np.array([ins_v, ins_num1[6:6 + 10]]).T), key=(lambda x: x[0])))[:, 1].cuda()
ins_t = ins_num[6 + 10:6 + 10 + 15]
ins_t_num = torch.tensor(sorted(list(np.array([ins_t, ins_num1[6 + 10:6 + 10 + 15]]).T), key=(lambda x: x[0])))[:,
            1].cuda()
ins_ivt = ins_num[6 + 10 + 15:]
ins_ivt_num = torch.tensor(sorted(list(np.array([ins_ivt, ins_num1[6 + 10 + 15:]]).T), key=(lambda x: x[0])))[:,
              1].cuda()
FLAGS.ins_ivt_num = np.array(sorted(list(np.array([ins_ivt, ins_num1[6 + 10 + 15:]]).T), key=(lambda x: x[0])))[:,
                    1]
tail_classes_i = ins_ivt[-4:]
tail_classes_v = ins_ivt[-8:]
tail_classes_t = ins_ivt[-6:]
if len(FLAGS.tail_classes_ivt) == 0:
    FLAGS.true_tail_classes_ivt = ins_ivt[-1 * (FLAGS.tail_num):]
else:
    FLAGS.true_tail_classes_ivt = FLAGS.tail_classes_ivt
valid_c = [i for idx, i in enumerate(range(100)) if i not in FLAGS.drop_classes]
valid_c_dict = {i: idx for idx, i in enumerate(valid_c)}
# all_c_dict = {idx: idx for idx, i in enumerate(valid_c)}
tail_classes_ivt = [valid_c_dict[ta] for ta in FLAGS.true_tail_classes_ivt if ta not in FLAGS.drop_classes]
print(tail_classes_ivt)
total_video_loss = []


def creat_mask(num, head_list):
    tail_mask = torch.zeros((num,)).cuda()
    true_tail_classes_ivt = [i for i in range(num) if i not in head_list]
    tail_mask[true_tail_classes_ivt] = 1
    head_mask = torch.ones((num,)).cuda() - tail_mask
    return tail_mask, head_mask


FLAGS.tail_mask, FLAGS.head_mask = creat_mask(100, [17, 60, 19])
FLAGS.tail_mask_i, FLAGS.head_mask_i = creat_mask(6, [0, 2])
FLAGS.tail_mask_v, FLAGS.head_mask_v = creat_mask(10, [1, 2])
FLAGS.tail_mask_t, FLAGS.head_mask_t = creat_mask(15, [0, 8])

mAP = ivtmetrics.Recognition(100)
FLAGS.bank = mAP.bank
mAP.reset_global()


def train_loop(dataloader, model, activation, loss_fn_ivt, optimizers, scheduler,
               epoch, writer, final_eval=False):
    c_nums = [6, 10, 15]
    for batch, (img, (y1, y2, y3, y4), img_path) in enumerate(dataloader):
        if batch > len(dataloader) / FLAGS.train_div:
            break
        img1, img2, y1, y1_s, y2, y2_s, y3, y3_s, y4, y4_s = img[0].cuda(), img[1].cuda(), y1[0].cuda(), y1[1].cuda(), \
                                                             y2[0].cuda(), y2[1].cuda(), y3[0].cuda(), y3[1].cuda(), y4[
                                                                 0].cuda(), y4[1].cuda()
        model.train()
        label_dict = {100: y4, 6: y1, 10: y2, 15: y3}
        tail_ivt_labels = y4 * FLAGS.tail_mask.unsqueeze(0).repeat_interleave(y4.shape[0], dim=0)
        tail_i_labels = torch.zeros_like(tail_ivt_labels)[:, :6]
        tail_v_labels = torch.zeros_like(tail_ivt_labels)[:, :10]
        tail_t_labels = torch.zeros_like(tail_ivt_labels)[:, :15]
        idx = torch.where(tail_ivt_labels == 1)
        if len(idx[0]) == 0:
            continue
        tail_i_labels[(idx[0], model.bank[idx[-1], 1])] = 1
        tail_v_labels[(idx[0], model.bank[idx[-1], 2])] = 1
        tail_t_labels[(idx[0], model.bank[idx[-1], 3])] = 1
        total_step = (epoch) * len(dataloader) + batch + 1
        all_tail_labels = [torch.vstack([y, y]) for y in [tail_ivt_labels, tail_i_labels, tail_v_labels, tail_t_labels]]

        if FLAGS.mlp:
            logits, logits_proto, queue_labels, triplet = model(im_q=img1, im_k=img2,
                                                                labels=[tail_ivt_labels, tail_i_labels, tail_v_labels,
                                                                        tail_t_labels])
            if len(logits[0]) == 0:
                print('notail_continue')
                continue
        else:
            _, _, _, triplet = model(im_q=img1, labels=[tail_ivt_labels, tail_i_labels, tail_v_labels,
                                                        tail_t_labels])

        # direct i,v,t
        _, logit_ivt = triplet[0]
        _, logit_i = triplet[1]
        _, logit_v = triplet[2]
        _, logit_t = triplet[3]
        loss_i1 = criterion(logit_i, y1).mean()
        loss_v1 = criterion(logit_v, y2).mean()
        loss_t1 = criterion(logit_t, y3).mean()

        loss_cls1 = loss_i1 + loss_v1 + loss_t1

        # ivt --> i, v, t

        logit_comps = []
        for i in range(3):
            comp_logits = []
            for c in range(c_nums[i]):
                idxes = np.where(mAP.bank[:, i + 1] == c)[0]
                comp_logits.append(torch.max(logit_ivt[:, idxes], dim=-1).values)
            logit_comps.append(torch.stack(comp_logits).permute(1, 0))

        loss_i = criterion(logit_comps[0], y1).mean()
        loss_v = criterion(logit_comps[1], y2).mean()
        loss_t = criterion(logit_comps[2], y3).mean()
        loss_ivt = criterion(logit_ivt, y4).mean()
        loss_cls_ivt = loss_i + loss_v + loss_t + loss_ivt

        if not FLAGS.mlp:
            loss = loss_cls1 + loss_cls_ivt
            info_loss = {
                'loss_i': loss_i.item(),
                'loss_v': loss_v.item(),
                'loss_t': loss_t.item(),
                'loss_ivt': loss_ivt.item(),
                'loss_cls_ivt': loss_cls_ivt.item(),
                'loss_i1': loss_i1.item(),
                'loss_v1': loss_v1.item(),
                'loss_t1': loss_t1.item(),
                'loss_cls1': loss_cls1.item(),
                'loss': loss.item()
            }
        else:
            if epoch < FLAGS.w_epoch:
                loss_con = criterion_con(logits[0], logits[1], queue_labels[0])
                loss = loss_con * FLAGS.w_con + loss_cls1
                print(
                    'total:{:2f}---con:{:2f}---proto:{:2f}'.format(loss.item(), loss_con.item(),
                                                                   loss_cls1.item()))
                info_loss = {
                    'loss_i': loss_i.item(),
                    'loss_v': loss_v.item(),
                    'loss_t': loss_t.item(),
                    'loss_ivt': loss_ivt.item(),
                    'loss_cls_ivt': loss_cls_ivt.item(),
                    'loss_i1': loss_i1.item(),
                    'loss_v1': loss_v1.item(),
                    'loss_t1': loss_t1.item(),
                    'loss_cls1': loss_cls1.item(),
                    'loss_con': loss_con.item(),
                    'loss': loss.item()
                }
            else:
                loss_cls = loss_cls_ivt + loss_cls1

                loss_con = criterion_con(logits[0], logits[1], queue_labels[0])

                loss_i1_proto = criterion(logits_proto[0][0],
                                          F.one_hot(logits_proto[-1][0].squeeze(-1), num_classes=6)).mean()
                loss_v1_proto = criterion(logits_proto[0][1],
                                          F.one_hot(logits_proto[-1][1].squeeze(-1), num_classes=10)).mean()
                loss_t1_proto = criterion(logits_proto[0][2],
                                          F.one_hot(logits_proto[-1][2].squeeze(-1), num_classes=15)).mean()

                loss_con_proto1 = loss_i1_proto + loss_v1_proto + loss_t1_proto

                loss_tail_ivt = criterion(logits[-1], F.one_hot(logits[-2][:, 0], num_classes=100)).mean()

                loss = loss_cls + loss_con * FLAGS.w_con + loss_con_proto1 * FLAGS.w_proto + loss_tail_ivt * FLAGS.w_tail
                print(
                    'total:{:2f}---con:{:2f}---proto1:{:2f}---tail:{:2f}'.format(loss.item(), loss_con.item(),
                                                                                 loss_con_proto1.item(),
                                                                                 loss_tail_ivt.item()))
                info_loss = {
                    'loss_i': loss_i.item(),
                    'loss_v': loss_v.item(),
                    'loss_t': loss_t.item(),
                    'loss_ivt': loss_ivt.item(),
                    'loss_cls_ivt': loss_cls_ivt.item(),
                    'loss_tail_ivt': loss_tail_ivt.item(),
                    'loss_i1': loss_i1.item(),
                    'loss_v1': loss_v1.item(),
                    'loss_t1': loss_t1.item(),
                    'loss_cls1': loss_cls1.item(),
                    'loss_con': loss_con.item(),
                    'loss_i1_proto': loss_i1_proto.item(),
                    'loss_v1_proto': loss_v1_proto.item(),
                    'loss_t1_proto': loss_t1_proto.item(),
                    'loss_con_proto1': loss_con_proto1.item(),
                    'loss': loss.item()
                }

        writer.add_scalars('train/loss', info_loss, total_step)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        info_lr = {
            'lr_ivt': optimizers[0].optimizer.state_dict()['param_groups'][0]['lr'],
        }
        writer.add_scalars('train/lr', info_lr, total_step)
        for opt in optimizers:
            opt.optimizer.step()

    for sch in scheduler:
        sch.step()


def test_loop(dataloader, model, activation, writer, final_eval=False, mode='val'):
    global allstep, all_val_step
    mAP.reset()
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()
    c_nums = [6, 10, 15]

    with torch.no_grad():
        video_loss = 0
        for batch, (img, (y1, y2, y3, y4), img_path) in enumerate(dataloader):
            img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            model.eval()
            _, _, _, triplet = model(im_q=img)
            _, logit_ivt = triplet[0]
            logit_comps = []
            for i in range(3):
                comp_logits = []
                for c in range(c_nums[i]):
                    idxes = np.where(mAP.bank[:, i + 1] == c)[0]
                    comp_logits.append(torch.max(logit_ivt[:, idxes], dim=-1).values)
                logit_comps.append(torch.stack(comp_logits).permute(1, 0))

            loss_i = criterion(logit_comps[0], y1).mean()
            loss_v = criterion(logit_comps[1], y2).mean()
            loss_t = criterion(logit_comps[2], y3).mean()
            loss_ivt = criterion(logit_ivt, y4).mean()

            loss = loss_i + loss_v + loss_t + loss_ivt
            video_loss = video_loss + loss

            info_loss = {
                'loss_i': loss_i.item(),
                'loss_v': loss_v.item(),
                'loss_t': loss_t.item(),
                'loss_ivt': loss_ivt.item(),
                'loss': loss.item()
            }
            restore_y4 = torch.zeros((len(y4), 100))
            restore_logit_ivt = torch.zeros((len(y4), 100))
            for dc in range(100):
                if dc not in FLAGS.drop_classes:
                    restore_y4[:, dc] = y4.float().detach().cpu()[:, valid_c_dict[dc]]
                    restore_logit_ivt[:, dc] = activation(logit_ivt).detach().cpu()[:, valid_c_dict[dc]]
            mAPi.update(y1.float().detach().cpu(),
                        activation(logit_comps[0]).detach().cpu())  # Log metrics
            mAPv.update(y2.float().detach().cpu(),
                        activation(logit_comps[1]).detach().cpu())  # Log metrics
            mAPt.update(y3.float().detach().cpu(),
                        activation(logit_comps[2]).detach().cpu())  # Log metrics
            mAP.update(restore_y4,
                       restore_logit_ivt)  # Log metrics
            if mode == 'test':
                if final_eval:
                    total_step = allstep + 1
                    writer.add_scalars('final_test/loss', info_loss, total_step)
                else:
                    total_step = allstep + 1
                    writer.add_scalars('test/loss', info_loss, total_step)
                allstep = allstep + 1
            else:
                if final_eval:
                    total_step = all_val_step + 1
                    writer.add_scalars('final_val/loss', info_loss, total_step)
                    all_val_step = all_val_step + 1
                else:
                    total_step = all_val_step + 1
                    writer.add_scalars('val/loss', info_loss, total_step)
                    all_val_step = all_val_step + 1
        video_loss = video_loss / len(dataloader)
        total_video_loss.append(video_loss)

    mAP.video_end()
    mAPv.video_end()
    mAPt.video_end()
    mAPi.video_end()


def weight_mgt(score, epoch):
    # hyperparameter selection based on validation set
    global benchmark
    torch.save(model.state_dict(), ckpt_path_epoch + '_latest.pth')
    if epoch == FLAGS.w_epoch - 1:
        torch.save(model.state_dict(), ckpt_path_epoch + '_{}.pth'.format(str(FLAGS.w_epoch)))
    if score > benchmark.item():
        torch.save(model.state_dict(), ckpt_path)
        benchmark = score
        print(f'>>> Saving checkpoint for epoch {epoch + 1} at {ckpt_path}, time {time.ctime()} ',
              file=open(logfile, 'a+'))
        return "increased"
    else:
        return "decreased"


# %% checkpoints/weights
def load_model(model, dir):
    pretrained_dict = torch.load(dir)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# Or constant weights from average of the random sampling of the dataset: we found this to produce better result.
if FLAGS.pos_w == 'ones':
    tool_weight = [1 for i in range(6)]
    verb_weight = [1 for i in range(10)]
    target_weight = [1 for i in range(15)]
elif FLAGS.pos_w == 'pre_w':
    tool_weight = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
    verb_weight = [0.60002400, 0.60002400, 0.60002400, 0.61682467, 0.67082683, 0.80163207, 0.70562823, 2.11208448,
                   2.69230769, 0.60062402]
    target_weight = [0.49752894, 0.52041527, 0.49752894, 0.51394739, 2.71899565, 1.75577963, 0.58509403, 1.25228034,
                     0.49752894, 2.42993134, 0.49802647, 0.87266576, 1.36074165, 0.50150917, 0.49802647]

# %% model
if FLAGS.loss_type == 'i':
    FLAGS.num_class = 6
elif FLAGS.loss_type == 'v':
    FLAGS.num_class = 10
elif FLAGS.loss_type == 't':
    FLAGS.num_class = 15
else:
    FLAGS.num_class = 100 - len(FLAGS.drop_classes)

model = network.build_q2l(FLAGS)
model = model.cuda()

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('BackBone: Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
# %% performance tracker for hp tuning
benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
print("Model built ...")
allstep = 0
all_val_step = 0
# %% Loss
activation = nn.Sigmoid()
loss_fn_i = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).cuda())
loss_fn_v = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).cuda())
loss_fn_t = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).cuda())
loss_fn_ivt = nn.BCEWithLogitsLoss()
criterion_ce = nn.CrossEntropyLoss()
# criterion
criterion = models.aslloss.AsymmetricLossOptimized(
    gamma_neg=2, gamma_pos=0,
    clip=0,
    disable_torch_grad_focal_loss=True,
    eps=1e-5,
)
from loss import *

criterion_con = KCL(args=FLAGS, K=FLAGS.moco_k, k=7, temperature=FLAGS.moco_t).cuda()
criterion_con_proto = KCLProto(args=FLAGS, K=FLAGS.moco_k, k=7, temperature=FLAGS.moco_t).cuda()
criterion_con_rank = SupConLoss_rank(K=FLAGS.moco_k, temperature=FLAGS.moco_t).cuda()
# %% evaluation metrics

mAPi = ivtmetrics.Recognition(6)
mAPv = ivtmetrics.Recognition(10)
mAPt = ivtmetrics.Recognition(15)
mAPi.reset_global()
mAPv.reset_global()
mAPt.reset_global()
print("Metrics built ...")

# %% optimizer and lr scheduler
wp_lr = [lr / power for lr in learning_rates]

optimizer_ivt = torch.optim.SGD(model.parameters(), lr=wp_lr[2], weight_decay=weight_decay)
scheduler_ivta = torch.optim.lr_scheduler.LinearLR(optimizer_ivt, start_factor=power, total_iters=warmups[2])
scheduler_ivtb = torch.optim.lr_scheduler.ExponentialLR(optimizer_ivt, gamma=decay_rate)
scheduler_ivt = torch.optim.lr_scheduler.SequentialLR(optimizer_ivt, schedulers=[scheduler_ivta, scheduler_ivtb],
                                                      milestones=[warmups[2] + 1])


optimizer_ivt = SGD(optimizer=optimizer_ivt, model=model)

lr_schedulers = [scheduler_ivt]
optimizers = [optimizer_ivt]

print("Model's weight loaded ...")

dataset = dataloader.CholecT50(
    FLAG=FLAGS,
    dataset_dir=data_dir,
    dataset_variant=dataset_variant,
    test_fold=kfold,
    augmentation_list=data_augmentations,
)

# build dataset
train_dataset, val_dataset, test_dataset, test_train_dataset = dataset.build()

# train and val data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=3 * batch_size,
                              num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)

val_dataloaders = []
for video_dataset in val_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=3 * batch_size,
                                 num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
    val_dataloaders.append(test_dataloader)

# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=3 * batch_size,
                                 num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
    test_dataloaders.append(test_dataloader)
test_train_dataloaders = []
for video_dataset in test_train_dataset:
    test_train_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False,
                                       prefetch_factor=3 * batch_size,
                                       num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
    test_train_dataloaders.append(test_train_dataloader)
print("Dataset loaded ...")

# %% log config
header1 = "** Run: {} | Framework: PyTorch | Method: {} | Version: {} | Data: CholecT50 | Batch: {} **".format(
    os.path.basename(__file__), modelname, version, batch_size)
header2 = "** Time: {} | Start: {}-epoch  {}-steps | Init CKPT: {} | Save CKPT: {} **".format(time.ctime(), 0, 0,
                                                                                              resume_ckpt, ckpt_path)
header3 = "** LR Config: Init: {} | Peak: {} | Warmup Epoch: {} | Rise: {} | Decay {} | train params {} | all params {} **".format(
    [float(f"{op.optimizer.state_dict()['param_groups'][0]['lr']:.6f}") for op in optimizers],
    [float(f'{v:.6f}') for v in wp_lr],
    warmups, power,
    decay_rate, pytorch_train_params, pytorch_total_params)
maxlen = max(len(header1), len(header2), len(header3))
# maxlen = max(len(header1), len(header2))
header1 = "{}{}{}".format('*' * ((maxlen - len(header1)) // 2 + 1), header1, '*' * ((maxlen - len(header1)) // 2 + 1))
header2 = "{}{}{}".format('*' * ((maxlen - len(header2)) // 2 + 1), header2, '*' * ((maxlen - len(header2)) // 2 + 1))
header3 = "{}{}{}".format('*' * ((maxlen - len(header3)) // 2 + 1), header3, '*' * ((maxlen - len(header3)) // 2 + 1))
maxlen = max(len(header1), len(header2), len(header3))
# maxlen = max(len(header1), len(header2))
writer = SummaryWriter(model_dir)

print("\n\n\n{}\n{}\n{}\n{}\n{}".format("*" * maxlen, header1, header2, header3, "*" * maxlen),
      file=open(logfile, 'a+'))
print("Experiment started ...\n   logging outputs to: ", logfile)

# %% run
if is_train:
    for epoch in range(0, epochs):
        try:

            print("Traning | lr: {} | epoch {}".format(
                [op.optimizer.state_dict()['param_groups'][0]['lr'] for op in optimizers],
                epoch), end=" | ",
                file=open(logfile, 'a+'))
            torch.cuda.empty_cache()

            train_loop(train_dataloader, model, activation, loss_fn_ivt, optimizers,
                       lr_schedulers, epoch, writer)

            # val
            # if (epoch + 1) % val_interval == 0:
            if (epoch + 1) % val_interval == 0 or epoch == FLAGS.w_epoch - 1:
                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("Evaluating @ epoch: ", epoch, file=open(logfile, 'a+'))
                total_video_loss = []
                for i, val_dataloader in enumerate(val_dataloaders):
                    test_loop(val_dataloader, model, activation, writer, final_eval=False)
                total_v_loss = 0
                for l in total_video_loss:
                    total_v_loss = total_v_loss + l
                total_v_loss = total_v_loss / len(total_video_loss)
                info_loss = {
                    'loss_ivt': total_v_loss.item(),

                }
                writer.add_scalars('val/loss_vid', info_loss, epoch)
                if FLAGS.loss_type == 'i':
                    behaviour = weight_mgt(mAPi.compute_video_AP()['mAP'], epoch=epoch)
                elif FLAGS.loss_type == 'v':
                    behaviour = weight_mgt(mAPv.compute_video_AP()['mAP'], epoch=epoch)
                elif FLAGS.loss_type == 't':
                    behaviour = weight_mgt(mAPt.compute_video_AP()['mAP'], epoch=epoch)
                else:
                    behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                if FLAGS.loss_type in ['i', 'v', 't']:
                    mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
                else:
                    mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
                    mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
                    mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

                mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
                mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
                mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
                info_mAP = {
                    'mAP_i': mAP_i["mAP"],
                    'mAP_v': mAP_v["mAP"],
                    'mAP_t': mAP_t["mAP"],
                    'mAP_iv': mAP_iv["mAP"],
                    'mAP_it': mAP_it["mAP"],
                    'mAP_ivt': mAP_ivt["mAP"],
                }
                writer.add_scalars('val/mAP', info_mAP, epoch)
                print(
                    "\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP => ivt: [{:.5f}] ".format((time.time() - start),
                                                                                                 info_mAP[
                                                                                                     'mAP_ivt']),
                    file=open(logfile, 'a+'))

                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("Test @ epoch: ", epoch, file=open(logfile, 'a+'))
                total_video_loss = []
                for i, test_dataloader in enumerate(test_dataloaders):
                    test_loop(test_dataloader, model, activation, writer, final_eval=False, mode='test')
                total_v_loss = 0
                for l in total_video_loss:
                    total_v_loss = total_v_loss + l
                total_v_loss = total_v_loss / len(total_video_loss)
                info_loss = {
                    'loss_ivt': total_v_loss.item(),

                }
                writer.add_scalars('test/loss_vid', info_loss, epoch)
                # behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                if FLAGS.loss_type in ['i', 'v', 't']:
                    mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
                else:
                    mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
                    mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
                    mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

                mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
                mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
                mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
                info_mAP = {
                    'mAP_i': mAP_i["mAP"],
                    'mAP_v': mAP_v["mAP"],
                    'mAP_t': mAP_t["mAP"],
                    'mAP_iv': mAP_iv["mAP"],
                    'mAP_it': mAP_it["mAP"],
                    'mAP_ivt': mAP_ivt["mAP"],
                }
                writer.add_scalars('test/mAP', info_mAP, epoch)
                print(
                    "\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP => ivt: [{:.5f}] ".format((time.time() - start),
                                                                                                 info_mAP[
                                                                                                     'mAP_ivt']),
                    file=open(logfile, 'a+'))

                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("TestTrain @ epoch: ", epoch, file=open(logfile, 'a+'))
                total_video_loss = []
                for i, test_dataloader in enumerate(test_train_dataloaders):
                    test_loop(test_dataloader, model, activation, writer, final_eval=False, mode='test')
                total_v_loss = 0
                for l in total_video_loss:
                    total_v_loss = total_v_loss + l
                total_v_loss = total_v_loss / len(total_video_loss)
                info_loss = {
                    'loss_ivt': total_v_loss.item(),

                }
                writer.add_scalars('train/loss_vid', info_loss, epoch)
                # behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                if FLAGS.loss_type in ['i', 'v', 't']:
                    mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
                else:
                    mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
                    mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
                    mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

                mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
                mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
                mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
                info_mAP = {
                    'mAP_i': mAP_i["mAP"],
                    'mAP_v': mAP_v["mAP"],
                    'mAP_t': mAP_t["mAP"],
                    'mAP_iv': mAP_iv["mAP"],
                    'mAP_it': mAP_it["mAP"],
                    'mAP_ivt': mAP_ivt["mAP"],
                }
                writer.add_scalars('train/mAP', info_mAP, epoch)
                print(
                    "\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP => ivt: [{:.5f}] ".format((time.time() - start),
                                                                                                 info_mAP[
                                                                                                     'mAP_ivt']),
                    file=open(logfile, 'a+'))
        except KeyboardInterrupt:
            print(f'>> Process cancelled by user at {time.ctime()}, ...', file=open(logfile, 'a+'))
            sys.exit(1)
    test_ckpt = ckpt_path

# %% eval
if is_test:
    print("Test weight: ", test_ckpt)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if not is_train:
        test_list = ['best']
    else:
        test_list = ['latest', 'best']
    for tag in test_list:
        print('========', tag, '==============', file=open(logfile, 'a+'))
        if tag == 'best':
            model.load_state_dict(torch.load(test_ckpt), strict=False)
        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0

        print('========testdata==============', file=open(logfile, 'a+'))
        for test_dataloader in test_dataloaders:
            test_loop(test_dataloader, model, activation, writer, final_eval=True, mode='test')

        mAPs = {'ivt': mAP, 'i': mAPi, 'v': mAPv, 't': mAPt}
        import pickle

        f = open(model_dir + '/mAPs_test_k' + str(kfold) + str(tag) + '.pckl', 'wb')
        pickle.dump(mAPs, f)
        f.close()
        mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_i_ivt = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
        mAP_v_ivt = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
        mAP_t_ivt = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

        mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
        mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
        mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
        print('-' * 50, file=open(logfile, 'a+'))
        print('SingleTest Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t["AP"]}', file=open(logfile, 'a+'))
        print('IVTTest Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'IV  : {mAP_iv["AP"]}', file=open(logfile, 'a+'))
        print(f'IT  : {mAP_it["AP"]}', file=open(logfile, 'a+'))
        print(f'IVT : {mAP_ivt["AP"]}', file=open(logfile, 'a+'))
        print('-' * 50, file=open(logfile, 'a+'))

        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i_ivt["mAP"]:.4f} | {mAP_v_ivt["mAP"]:.4f} | {mAP_t_ivt["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        print('------------singletest-------------', file=open(logfile, 'a+'))
        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i["mAP"]:.4f} | {mAP_v["mAP"]:.4f} | {mAP_t["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        top5 = [mAP.topK(5, 'i'), mAP.topK(5, 'v'), mAP.topK(5, 't'), mAP.topK(5, 'iv'), mAP.topK(5, 'it'),
                mAP.topK(5, 'ivt')]
        top10 = [mAP.topK(10, 'i'), mAP.topK(10, 'v'), mAP.topK(10, 't'), mAP.topK(10, 'iv'), mAP.topK(10, 'it'),
                 mAP.topK(10, 'ivt')]
        top20 = [mAP.topK(20, 'i'), mAP.topK(20, 'v'), mAP.topK(20, 't'), mAP.topK(20, 'iv'), mAP.topK(20, 'it'),
                 mAP.topK(20, 'ivt')]
        print(f'top 5:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top5[0]:.4f} | {top5[1]:.4f} | {top5[2]:.4f} | {top5[3]:.4f} | {top5[4]:.4f} | {top5[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 10:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top10[0]:.4f} | {top10[1]:.4f} | {top10[2]:.4f} | {top10[3]:.4f} | {top10[4]:.4f} | {top10[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 20:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top20[0]:.4f} | {top20[1]:.4f} | {top20[2]:.4f} | {top20[3]:.4f} | {top20[4]:.4f} | {top20[5]:.4f} ',
            file=open(logfile, 'a+'))
        print('=' * 50, file=open(logfile, 'a+'))

        print('========testdata==============', file=open(logfile, 'a+'))
        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        for test_dataloader in val_dataloaders:
            test_loop(test_dataloader, model, activation, writer, final_eval=True, mode='test')

        mAPs = {'ivt': mAP, 'i': mAPi, 'v': mAPv, 't': mAPt}
        import pickle

        f = open(model_dir + '/mAPs_val_k' + str(kfold) + str(tag) + '.pckl', 'wb')
        pickle.dump(mAPs, f)
        f.close()
        mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_i_ivt = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
        mAP_v_ivt = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
        mAP_t_ivt = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

        mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
        mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
        mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
        print('-' * 50, file=open(logfile, 'a+'))
        print('SingleTest Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t["AP"]}', file=open(logfile, 'a+'))
        print('IVTTest Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'IV  : {mAP_iv["AP"]}', file=open(logfile, 'a+'))
        print(f'IT  : {mAP_it["AP"]}', file=open(logfile, 'a+'))
        print(f'IVT : {mAP_ivt["AP"]}', file=open(logfile, 'a+'))
        print('-' * 50, file=open(logfile, 'a+'))

        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i_ivt["mAP"]:.4f} | {mAP_v_ivt["mAP"]:.4f} | {mAP_t_ivt["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        print('------------singletest-------------', file=open(logfile, 'a+'))
        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i["mAP"]:.4f} | {mAP_v["mAP"]:.4f} | {mAP_t["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        top5 = [mAP.topK(5, 'i'), mAP.topK(5, 'v'), mAP.topK(5, 't'), mAP.topK(5, 'iv'), mAP.topK(5, 'it'),
                mAP.topK(5, 'ivt')]
        top10 = [mAP.topK(10, 'i'), mAP.topK(10, 'v'), mAP.topK(10, 't'), mAP.topK(10, 'iv'), mAP.topK(10, 'it'),
                 mAP.topK(10, 'ivt')]
        top20 = [mAP.topK(20, 'i'), mAP.topK(20, 'v'), mAP.topK(20, 't'), mAP.topK(20, 'iv'), mAP.topK(20, 'it'),
                 mAP.topK(20, 'ivt')]
        print(f'top 5:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top5[0]:.4f} | {top5[1]:.4f} | {top5[2]:.4f} | {top5[3]:.4f} | {top5[4]:.4f} | {top5[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 10:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top10[0]:.4f} | {top10[1]:.4f} | {top10[2]:.4f} | {top10[3]:.4f} | {top10[4]:.4f} | {top10[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 20:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top20[0]:.4f} | {top20[1]:.4f} | {top20[2]:.4f} | {top20[3]:.4f} | {top20[4]:.4f} | {top20[5]:.4f} ',
            file=open(logfile, 'a+'))
        print('=' * 50, file=open(logfile, 'a+'))

        print('========traindata==============', file=open(logfile, 'a+'))
        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        for test_dataloader in test_train_dataloaders:
            test_loop(test_dataloader, model, activation, writer, final_eval=True, mode='test')

        mAPs = {'ivt': mAP, 'i': mAPi, 'v': mAPv, 't': mAPt}
        import pickle

        f = open(model_dir + '/mAPs_testTrain_k' + str(kfold) + str(tag) + '.pckl', 'wb')
        pickle.dump(mAPs, f)
        f.close()
        mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_i_ivt = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
        mAP_v_ivt = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
        mAP_t_ivt = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

        mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
        mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
        mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
        print('-' * 50, file=open(logfile, 'a+'))
        print('SingleTest Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t["AP"]}', file=open(logfile, 'a+'))
        print('IVTTest Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t_ivt["AP"]}', file=open(logfile, 'a+'))
        print(f'IV  : {mAP_iv["AP"]}', file=open(logfile, 'a+'))
        print(f'IT  : {mAP_it["AP"]}', file=open(logfile, 'a+'))
        print(f'IVT : {mAP_ivt["AP"]}', file=open(logfile, 'a+'))
        print('-' * 50, file=open(logfile, 'a+'))

        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i_ivt["mAP"]:.4f} | {mAP_v_ivt["mAP"]:.4f} | {mAP_t_ivt["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        print('------------singletest-------------', file=open(logfile, 'a+'))
        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i["mAP"]:.4f} | {mAP_v["mAP"]:.4f} | {mAP_t["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        top5 = [mAP.topK(5, 'i'), mAP.topK(5, 'v'), mAP.topK(5, 't'), mAP.topK(5, 'iv'), mAP.topK(5, 'it'),
                mAP.topK(5, 'ivt')]
        top10 = [mAP.topK(10, 'i'), mAP.topK(10, 'v'), mAP.topK(10, 't'), mAP.topK(10, 'iv'), mAP.topK(10, 'it'),
                 mAP.topK(10, 'ivt')]
        top20 = [mAP.topK(20, 'i'), mAP.topK(20, 'v'), mAP.topK(20, 't'), mAP.topK(20, 'iv'), mAP.topK(20, 'it'),
                 mAP.topK(20, 'ivt')]
        print(f'top 5:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top5[0]:.4f} | {top5[1]:.4f} | {top5[2]:.4f} | {top5[3]:.4f} | {top5[4]:.4f} | {top5[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 10:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top10[0]:.4f} | {top10[1]:.4f} | {top10[2]:.4f} | {top10[3]:.4f} | {top10[4]:.4f} | {top10[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 20:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top20[0]:.4f} | {top20[1]:.4f} | {top20[2]:.4f} | {top20[3]:.4f} | {top20[4]:.4f} | {top20[5]:.4f} ',
            file=open(logfile, 'a+'))
        print('=' * 50, file=open(logfile, 'a+'))

# %% End
print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-" * maxlen),
      file=open(logfile, 'a+'))
