#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import libraries
import copy
import os
import sys
import time
import torch
import random
import network
import argparse
import platform
import ivtmetrics  # You must "pip install ivtmetrics" to use
import dataloader_test as dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import copy

np.seterr(invalid='ignore')
# %% @args parsing
# %% @args parsing
parser = argparse.ArgumentParser()
# model
parser.add_argument('--model', type=str, default='rendezvous', choices=['rendezvous'], help='Model name?')
parser.add_argument('--version', type=str, default='', help='Model version control (for keeping several versions)')
parser.add_argument('--version1', type=str, default='', help='Model version control (for keeping several versions)')
parser.add_argument('--hr_output', action='store_true',
                    help='Whether to use higher resolution output (32x56) or not (8x14). Default: False')
parser.add_argument('--use_ln', action='store_true',
                    help='Whether to use layer norm or batch norm in AddNorm() function. Default: False')
parser.add_argument('--decoder_layer', type=int, default=8, help='Number of MHMA layers ')
# job
parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
parser.add_argument('--input_dim', type=int, default=1536, help='seed for initializing training.')
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test', action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,
                    help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/home/shuangchun/Data/Video/CholecT45/CholecT45',
                    help='path to dataset?')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval',
                    choices=['cholect50', 'cholect45', 'cholect50-challenge', 'cholect50-crossval',
                             'cholect45-crossval', 'cholect45-challenge'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1, choices=[1, 2, 3, 4, 5, ],
                    help='The test split in k-fold cross-validation')
parser.add_argument('--image_width', type=int, default=448, help='Image width ')
parser.add_argument('--image_height', type=int, default=256, help='Image height ')
parser.add_argument('--image_channel', type=int, default=3, help='Image channels ')
parser.add_argument('--num_tool_classes', type=int, default=6, help='Number of tool categories')
parser.add_argument('--final_embedding_dim', type=int, default=512, help='Number of tool categories')
parser.add_argument('--num_verb_classes', type=int, default=10, help='Number of verb categories')
parser.add_argument('--num_target_classes', type=int, default=15, help='Number of target categories')
parser.add_argument('--num_triplet_classes', type=int, default=100, help='Number of triplet categories')
parser.add_argument('--augmentation_list', type=str, nargs='*',
                    default=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                    help='List augumentation styles (see dataloader.py for list of supported styles).')
# hp
parser.add_argument('--loss_type', type=str, default='t', help='path to pretrain_weight?')
parser.add_argument('-b', '--batch', type=int, default=32, help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=100, help='How many training epochs?')
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[9, 18, 58],
                    help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01, 0.01, 0.01],
                    help='List learning rates for tool, verb-target, triplet respectively')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization weight decay constant')
parser.add_argument('--decay_steps', type=int, default=10, help='Step to exponentially decay')
parser.add_argument('--in_feat_dim', type=int, default=1536, help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99, help='Learning rates weight decay rate')
parser.add_argument('--momentum', type=float, default=0.95, help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1, help='Learning rates weight decay power')
# weights
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
# device
parser.add_argument('--gpu', type=str, default="0,1,2",
                    help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')
FLAGS, unparsed = parser.parse_known_args()

random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)
# %% @params definitions
version1 = copy.deepcopy(FLAGS.version)
if FLAGS.loss_type != 'all':
    FLAGS.version = FLAGS.version + '_' + FLAGS.loss_type
is_train = FLAGS.train
is_test = FLAGS.test
dataset_variant = FLAGS.dataset_variant
data_dir = FLAGS.data_dir
kfold = FLAGS.kfold if "crossval" in dataset_variant else 0
version = FLAGS.version
hr_output = FLAGS.hr_output
use_ln = FLAGS.use_ln
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
set_chlg_eval = True if "challenge" in dataset_variant else False  # To observe challenge evaluation protocol
gpu = ",".join(str(FLAGS.gpu).split(","))
decodelayer = FLAGS.decoder_layer
addnorm = "layer" if use_ln else "batch"
modelsize = "high" if hr_output else "low"
FLAGS.multigpu = len(gpu) > 1  # not yet implemented !
mheaders = ["", "l", "cholect", "k"]
margs = [FLAGS.model, decodelayer, dataset_variant, kfold]
wheaders = ["norm", "res"]
wargs = [addnorm, modelsize]
modelname = "_".join(["{}{}".format(x, y) for x, y in zip(mheaders, margs) if len(str(y))]) + "_" + \
            "_".join(["{}{}".format(x, y) for x, y in zip(wargs, wheaders) if len(str(x))])
model_dir = "./__checkpoint__/run_{}".format(version)
if not os.path.exists(model_dir): os.makedirs(model_dir)
resume_ckpt = None
ckpt_path = os.path.join(model_dir, '{}.pth'.format(modelname))
ckpt_path_epoch = os.path.join(model_dir, '{}'.format(modelname))
logfile = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations = FLAGS.augmentation_list
iterable_augmentations = []
print("Configuring network ...")

def assign_gpu(gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def test_loop(dataloader, model):
    mAP.reset()
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()
    feats = []

    with torch.no_grad():
        for batch, (img, (y1, y2, y3, y4), paths) in enumerate(dataloader):
            img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            model.eval()
            img = img.transpose(0, 1).unsqueeze(0)
            (tool, _), (verb, _), (target, _), (triplet, feat) = model(img)
            feats.append(feat.permute(0, 2, 1).squeeze(0))
            logit_i = tool[0]
            logit_v = verb[0]
            logit_t = target[0]
            logit_ivt = triplet[0]
            mAPi.update(y1.float().detach().cpu(), logit_i.detach().cpu())  # Log metrics
            mAPv.update(y2.float().detach().cpu(), logit_v.detach().cpu())  # Log metrics
            mAPt.update(y3.float().detach().cpu(), logit_t.detach().cpu())  # Log metrics
            mAP.update(y4.float().detach().cpu(), logit_ivt.detach().cpu())  # Log metrics
        feats = torch.vstack(feats)

    mAP.video_end()
    mAPv.video_end()
    mAPt.video_end()
    mAPi.video_end()
    return feats


def weight_mgt(score, epoch):
    # hyperparameter selection based on validation set
    global benchmark
    torch.save(model.state_dict(), ckpt_path_epoch + 'latest.pth')
    if score > benchmark.item():
        torch.save(model.state_dict(), ckpt_path)
        benchmark = score
        print(f'>>> Saving checkpoint for epoch {epoch + 1} at {ckpt_path}, time {time.ctime()} ',
              file=open(logfile, 'a+'))
        return "increased"
    else:
        return "decreased"


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


def load_model(model, dir):
    print('lode_dir', dir)
    pretrained_dict = torch.load(dir)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


inter_channels = [256, 384, 576, 864]
num_block = 2
head = 8
mlp_ratio = 8
in_feat_dim = FLAGS.in_feat_dim

final_embedding_dim = FLAGS.final_embedding_dim
model = network.VideoNas(FLAGS, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim).cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('MS-TCT: Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
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

# %% evaluation metrics
mAP = ivtmetrics.Recognition(100)
mAP.reset_global()
mAP_ivt = ivtmetrics.Recognition(100)
mAP_ivt.reset_global()
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

lr_schedulers = [scheduler_ivt]
optimizers = [optimizer_ivt]

print("Model's weight loaded ...")
# %% data loading : variant and split selection (Note: original paper used different augumentation per epoch)

dataset = dataloader.CholecT50(
    dataset_dir=data_dir,
    dataset_variant=dataset_variant,
    test_fold=kfold,
    augmentation_list=data_augmentations,
    args=FLAGS
)

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()

# train and val data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=10 * batch_size,
                              num_workers=10, pin_memory=True, persistent_workers=True, drop_last=False)
val_dataloaders = []
for video_dataset in val_dataset:
    val_dataloader = DataLoader(video_dataset, batch_size=256, shuffle=False,
                                prefetch_factor=10 * 256,
                                num_workers=10, pin_memory=True, persistent_workers=True, drop_last=False)
    val_dataloaders.append(val_dataloader)

test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=256, shuffle=False,
                                 prefetch_factor=3 * 256,
                                 num_workers=3, pin_memory=True, persistent_workers=True, drop_last=False)
    test_dataloaders.append(test_dataloader)
print("Dataset loaded ...")

# %% log config
header1 = "** Run: {} | Framework: PyTorch | Method: {} | Version: {} | Data: CholecT50 | Batch: {} **".format(
    os.path.basename(__file__), modelname, version, batch_size)
header2 = "** Time: {} | Start: {}-epoch  {}-steps | Init CKPT: {} | Save CKPT: {} **".format(time.ctime(), 0, 0,
                                                                                              resume_ckpt, ckpt_path)
header3 = "** LR Config: Init: {} | Peak: {} | Warmup Epoch: {} | Rise: {} | Decay {} | train params {} | all params {} **".format(
    [float(f"{op.state_dict()['param_groups'][0]['lr']:.6f}") for op in optimizers], [float(f'{v:.6f}') for v in wp_lr],
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
    print("Test weight: ", test_ckpt)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    test_list = ['latest']
    for tag in test_list:
        print('========', tag, '==============')
        print('========', tag, '==============', file=open(logfile, 'a+'))
        test_ckpt = './__checkpoint__/run_{}/rendezvous_l8_cholect{}_k{}_batchnorm_lowreslatest.pth'.format(
            FLAGS.version, FLAGS.dataset_variant, FLAGS.kfold)
        model.load_state_dict(torch.load(test_ckpt))

        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        all_feats = {}
        for test_dataloader in test_dataloaders:
            feats = test_loop(test_dataloader, model)
            all_feats[test_dataloader.dataset.img_dir[-2:]] = feats.detach().cpu().numpy()

        import pickle

        os.makedirs('../0-5fold/data_feats/run_{}'.format(version1), exist_ok=True)

        f = open(
            '../0-5fold/data_feats/run_{}/k{}_{}_feats.pkl'.format(version1, str(
                kfold), FLAGS.loss_type),
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
        f = open(
            '../0-5fold/data_feats/run_{}/k{}_{}_pred.pkl'.format(version1, str(
                kfold), FLAGS.loss_type),
            'wb')
        pickle.dump(results, f)
        f.close()

# %% End
print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-" * maxlen),
      file=open(logfile, 'a+'))
