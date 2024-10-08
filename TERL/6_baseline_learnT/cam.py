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
import cv2

from functools import partial
import sys

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
parser.add_argument('--val_interval', type=int, default=1,
                    help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
# parser.add_argument('--data_dir', type=str, default='/public/home/guisc/Data/Video/Surgical/CholecT45',
#                     help='path to dataset?')
# parser.add_argument('--data_dir', type=str, default='/SDIM/shuangchun/Data/Video/Surgical/CholecT45/CholecT45',
#                     help='path to dataset?')
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
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01, 0.01, 0.01],
                    help='List learning rates for tool, verb-target, triplet respectively')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization weight decay constant')
parser.add_argument('--decay_steps', type=int, default=10, help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99, help='Learning rates weight decay rate')
parser.add_argument('--momentum', type=float, default=0.95, help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1, help='Learning rates weight decay power')
# weights
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--loss_type', type=str, default='all', help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
parser.add_argument('--backbone', type=str, default='swin_L_384_22k', help='path to pretrain_weight?')
parser.add_argument('--img_size', type=int, default=384, help='path to pretrain_weight?')
parser.add_argument('--hidden_dim', type=int, default=1536, help='path to pretrain_weight?')

# device
parser.add_argument('--gpu', type=str, default="0,1,2",
                    help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')
FLAGS, unparsed = parser.parse_known_args()

# %% @params definitions
# seed
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)

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
set_chlg_eval = True if "challenge1" in dataset_variant else False  # To observe challenge evaluation protocol
gpu = ",".join(str(FLAGS.gpu).split(","))

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


def assign_gpu(gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def train_loop(dataloader, model, activation, loss_fn_ivt, optimizers, scheduler,
               epoch, writer, final_eval=False):
    for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
        img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
        model.train()

        tool, verb, target, triplet = model(img)
        cam_i, logit_i = tool
        cam_v, logit_v = verb
        cam_t, logit_t = target
        cam_ivt, logit_ivt = triplet
        # loss = criterion(logit_i, y1)
        loss_i = criterion(logit_i, y1)
        loss_v = criterion(logit_v, y2)
        loss_t = criterion(logit_t, y3)
        loss_ivt = criterion(logit_ivt, y4)
        # loss_i = loss_fn_i(logit_i, y1.float())
        # loss_v = loss_fn_v(logit_v, y2.float())
        # loss_t = loss_fn_t(logit_t, y3.float())
        # loss_ivt = loss_fn_ivt(logit_ivt, y4.float())
        # loss = (loss_i) + (loss_v) + (loss_t) + loss_ivt
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

        info_loss = {
            'loss_i': loss_i.item(),
            'loss_v': loss_v.item(),
            'loss_t': loss_t.item(),
            'loss_ivt': loss_ivt.item(),
            'loss': loss.item()
        }
        total_step = (epoch) * len(dataloader) + batch + 1
        writer.add_scalars('train/loss', info_loss, total_step)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        info_lr = {
            'lr_ivt': optimizers[0].state_dict()['param_groups'][0]['lr'],
        }
        writer.add_scalars('train/lr', info_lr, total_step)
        for opt in optimizers:
            opt.step()

    for sch in scheduler:
        sch.step()


def hook(module, input, output, feature_blob):
    feature_blob.append(output)


def compute_cam(activation, softmax_weight, class_ids):
    b, c, h, w = activation.shape
    cams = []
    for idx in class_ids:
        activation = activation.reshape(c, h * w)
        cam = softmax_weight[idx].dot(activation)
        cam = cam.reshape(h, w)
        # normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        # conver to [0, 255]
        cam = np.uint8(255 * cam)
        # reshape to (224, 224)
        cams.append(cv2.resize(cam, (256, 448)))

    return cams


def draw_CAM(model, input, img_path, y4):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    final_conv_layer = 'backbone'
    # output, features = model(img)

    # output = model.classifier(features)
    feature_blob = []
    model._modules.get(final_conv_layer).register_forward_hook(
        partial(hook, feature_blob=feature_blob))
    # model.basemodel._modules['basemodel']._modules.get(final_conv_layer).register_forward_hook(
    #     partial(hook, feature_blob=feature_blob))

    # get the softmax (last fc layer) weight
    params = list(model.parameters())
    softmax_weight = np.squeeze(params[-2].data.numpy())
    # input = load_image('./test.jpg')

    tool, verb, target, triplet = model(input)  # scores

    # labels = get_labels('./labels.json')
    cls = torch.where(y4 == 1)[-1].unique()

    prob = F.sigmoid(triplet[-1]).data.squeeze()

    prob, idx = prob.sort(0, descending=True)

    # print(labels)

    # output the top-5 prediction
    # for i in range(5):
    #     print('{:.3f} -> {}'.format(probs[i], labels[idx[i]]))

    # generate class activation map for the top-5 prediction
    if len(cls) == 0:
        cams = compute_cam(feature_blob[0], softmax_weight, idx[0:1])
    else:
        cams = compute_cam(feature_blob[0][0][0], softmax_weight, cls)

    # for i in range(len(cams)):
    # render cam and original image
    # print('output %s for the top-%s prediction: %s' % (filename, (i + 1), labels[idx[i]]))

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cams[0], (w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('003077_i.png', result)
    return result


def test_loop(dataloader, model, activation, writer, final_eval=False, mode='val'):
    global allstep, all_val_step
    mAP.reset()
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()

    with torch.no_grad():
        for batch, (img, (y1, y2, y3, y4), img_path) in enumerate(dataloader):
            # img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            model.eval()
            # tool, verb, target, triplet = model(img)
            for i in range(len(img)):
                if img_path[i].split('/')[-1] == '003077.png':
                    A = draw_CAM(model, img[i].unsqueeze(0), img_path[i], y1[i].unsqueeze(0))


def weight_mgt(score, epoch):
    # hyperparameter selection based on validation set
    global benchmark
    torch.save(model.state_dict(), ckpt_path_epoch + '_latest.pth')
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
    FLAGS.num_class = 100

model = network.build_q2l(FLAGS)
# model = model.cuda()
model = model
model = load_model(model, './pretrain/checkpoint.pkl')

# ema_m = network.ModelEma(model, 0.9997)  # 0.9997
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[FLAGS.gpu], broadcast_buffers=False)

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
# %% evaluation metrics
mAP = ivtmetrics.Recognition(100)
mAP.reset_global()
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

dataset = dataloader.CholecT50(
    FLAG=FLAGS,
    dataset_dir=data_dir,
    dataset_variant=dataset_variant,
    test_fold=kfold,
    augmentation_list=data_augmentations,
)

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()

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

# %% run
if is_train:
    for epoch in range(0, epochs):
        try:

            print("Traning | lr: {} | epoch {}".format([op.state_dict()['param_groups'][0]['lr'] for op in optimizers],
                                                       epoch), end=" | ",
                  file=open(logfile, 'a+'))
            torch.cuda.empty_cache()
            train_loop(train_dataloader, model, activation, loss_fn_ivt, optimizers,
                       lr_schedulers, epoch, writer)

            # val
            if (epoch + 1) % val_interval == 0:
                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("Evaluating @ epoch: ", epoch, file=open(logfile, 'a+'))
                for i, val_dataloader in enumerate(val_dataloaders):
                    test_loop(val_dataloader, model, activation, writer, final_eval=False)
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
                                                                                                     'mAP_' + FLAGS.loss_type]),
                    file=open(logfile, 'a+'))

                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("Test @ epoch: ", epoch, file=open(logfile, 'a+'))
                for i, test_dataloader in enumerate(test_dataloaders):
                    test_loop(test_dataloader, model, activation, writer, final_eval=False, mode='test')
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
                                                                                                     'mAP_' + FLAGS.loss_type]),
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
        print('========', tag, '==============')
        if tag == 'best':
            pre_dict = torch.load(test_ckpt)

            for k in list(pre_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                # print(k)

                if k not in model.state_dict().keys():
                    print(k)
                    del pre_dict[k]
            model.load_state_dict(pre_dict)

        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        for test_dataloader in test_dataloaders:
            test_loop(test_dataloader, model, activation, writer, final_eval=True, mode='test')

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
        print('Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t["AP"]}', file=open(logfile, 'a+'))
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
