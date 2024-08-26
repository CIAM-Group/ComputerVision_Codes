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
import dataloader_test
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
parser.add_argument('--version1', type=str, default='', help='Model version control (for keeping several versions)')

# job
parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test', action='store_true', help='to test')
parser.add_argument('--fix_backbone', action='store_true', help='to test')
parser.add_argument('--ht', action='store_true', help='to test')
parser.add_argument('--latest', action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,
                    help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/public/home/guisc/Data/Video/Surgical/CholecT45',
                    help='path to dataset?')
parser.add_argument('--rho', type=float, default=0.05)
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
                    help='List augumentation styles (see dataloader.py for list of supported styles).')
# hp
parser.add_argument('-b', '--batch', type=int, default=32, help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=100, help='How many training epochs?')
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
# %% @params definitions
# seed
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)


def assign_gpu(gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


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

FLAGS.bank_id = {6: 1, 10: 2, 15: 3, 100: 0}


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

def test_loop(dataloader, model, activation, writer, final_eval=False, mode='val'):
    global allstep, all_val_step
    mAP.reset()
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()
    c_nums = [6, 10, 15]
    feats = []

    with torch.no_grad():
        video_loss = 0
        for batch, (img, (y1, y2, y3, y4), img_path) in enumerate(dataloader):
            img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            model.eval()
            _, _, _, triplet = model(im_q=img)
            cam_ivt, logit_ivt = triplet[0]
            feats.append(cam_ivt)
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
            if FLAGS.loss_type == 'i':
                loss = loss_i
            elif FLAGS.loss_type == 'v':
                loss = loss_v
            elif FLAGS.loss_type == 't':
                loss = loss_t
            elif FLAGS.loss_type == 'ivt':
                loss = loss_ivt
            else:
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
        feats = torch.vstack(feats)

    mAP.video_end()
    mAPv.video_end()
    mAPt.video_end()
    mAPi.video_end()
    return feats


# %% checkpoints/weights
def load_model(model, dir):
    pretrained_dict = torch.load(dir)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# %% assign device and set debugger options
assign_gpu(gpu=gpu)
np.seterr(divide='ignore', invalid='ignore')
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Or constant weights from average of the random sampling of the dataset: we found this to produce better result.
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
# criterion
criterion = models.aslloss.AsymmetricLossOptimized(
    gamma_neg=2, gamma_pos=0,
    clip=0,
    disable_torch_grad_focal_loss=True,
    eps=1e-5,
)
from loss import *

criterion_con = KCL(args=FLAGS, K=FLAGS.moco_k, k=7, temperature=FLAGS.moco_t).cuda()
criterion_con_rank = SupConLoss_rank(K=FLAGS.moco_k, temperature=FLAGS.moco_t).cuda()

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
FLAGS.bank = mAP.bank

dataset = dataloader_test.CholecT50(
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


# %% eval
if is_test:
    if FLAGS.latest:
        test_ckpt = './__checkpoint__/run_' + FLAGS.version1 + '/rendezvous_l' + FLAGS.dataset_variant + '_cholect' + str(
            kfold) + '_latest.pth'
        FLAGS.version1 = FLAGS.version1 + '_latest'
    else:
        test_ckpt = './__checkpoint__/run_' + FLAGS.version1 + '/rendezvous_l' + FLAGS.dataset_variant + '_cholect' + str(
            kfold) + '.pth'
    print("Test weight: ", test_ckpt)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if not is_train:
        test_list = ['best']
    else:
        test_list = ['latest', 'best']
    for tag in test_list:
        print('========', tag, '==============')
        if tag == 'best':
            model.load_state_dict(torch.load(test_ckpt))
        else:
            model.load_state_dict(torch.load(test_ckpt[:-4] + '_latest.pth'))

        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        all_feats = {}
        for test_dataloader in test_dataloaders:
            feats = test_loop(test_dataloader, model, activation, writer, final_eval=True, mode='test')
            all_feats[test_dataloader.dataset.img_dir[-2:]] = feats.detach().cpu().numpy()

        import pickle

        os.makedirs('../0-5fold/data_feats/run_' + FLAGS.version1, exist_ok=True)

        f = open(
            '../0-5fold/data_feats/run_' + FLAGS.version1 + '/k' + str(
                kfold) + '_feats.pkl',
            'wb')
        pickle.dump(all_feats, f)
        f.close()
        if FLAGS.loss_type == 'i':
            curt_mAP = mAPi
        elif FLAGS.loss_type == 'v':
            curt_mAP = mAPv
        elif FLAGS.loss_type == 't':
            curt_mAP = mAPt
        else:
            curt_mAP = mAP
        results = {}
        for key, pred in zip(all_feats.keys(), curt_mAP.global_predictions):
            results[key] = pred
            assert len(results[key]) == len(all_feats[key])
        f = open('../0-5fold/data_feats/run_' + FLAGS.version1 + '/k' + str(
            kfold) + '_pred.pkl',
                 'wb')
        pickle.dump(results, f)
        f.close()

# %% End
print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-" * maxlen),
      file=open(logfile, 'a+'))
