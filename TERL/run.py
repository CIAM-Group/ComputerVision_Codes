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
import pickle

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
parser.add_argument('--latest', action='store_true', help='to test')
parser.add_argument('--use_ln', action='store_true',
                    help='Whether to use layer norm or batch norm in AddNorm() function. Default: False')
parser.add_argument('--decoder_layer', type=int, default=8, help='Number of MHMA layers ')
# job
parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test', action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,
                    help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/public/home/guisc/Data/Video/Surgical/CholecT45',
                    help='path to dataset?')
# parser.add_argument('--data_dir', type=str, default='/SDIM/shuangchun/Data/Video/Surgical/CholecT45/CholecT45',
#                     help='path to dataset?')
# parser.add_argument('--data_dir', type=str, default='/home/shuangchun/Data/Video/CholecT45/CholecT45',
#                     help='path to dataset?')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval',
                    choices=['cholect50', 'cholect45', 'cholect50-challenge', 'cholect50-crossval',
                             'cholect45-crossval', 'cholect45-challenge'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1,
                    help='The test split in k-fold cross-validation')
parser.add_argument('--train_div', type=int, default=1, help='path to pretrain_weight?')
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
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
parser.add_argument('--loss_type', type=str, default='i', help='path to pretrain_weight?')
parser.add_argument('--pos_w', type=str, default='pre_w', help='path to pretrain_weight?')
####ms-tcn2
parser.add_argument('--num_layers_PG', default="11", type=int)
parser.add_argument('--num_layers_R', default="10", type=int)
parser.add_argument('--num_R', default="3", type=int)

parser.add_argument('--fpn', action='store_true')
parser.add_argument('--mask', action='store_true')
parser.add_argument('--output', default=False, type=bool)
parser.add_argument('--feature', default=False, type=bool)
parser.add_argument('--trans', default=False, type=bool)
parser.add_argument('--prototype', default=False, type=bool)
parser.add_argument('--last', default=False, type=bool)
parser.add_argument('--first', default=False, type=bool)
parser.add_argument('--hier', default=False, type=bool)

##Transformer
parser.add_argument('--head_num', default=8)
parser.add_argument('--embed_num', default=512)
parser.add_argument('--input_dim', type=int, default=512)
parser.add_argument('--block_num', default=1)
parser.add_argument('--positional_encoding_type', default="learned", type=str, help="fixed or learned")

# device
parser.add_argument('--gpu', type=str, default="0,1,2",
                    help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')
FLAGS, unparsed = parser.parse_known_args()

random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)
# %% @params definitions
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
set_chlg_eval = True  # To observe challenge evaluation protocol
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
if FLAGS.latest:
    FLAGS.version1 = FLAGS.version1 + '_latest'
print("Configuring network ...")


# %% class weight balancing
def get_weight_balancing(case='cholect50'):
    # 50:   cholecT50, data splits as used in rendezvous paper
    # 50ch: cholecT50, data splits as used in CholecTriplet challenge
    # 45cv: cholecT45, official data splits (cross-val)
    # 50cv: cholecT50, official data splits (cross-val)
    switcher = {
        'cholect50': {
            'tool': [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455],
            'verb': [0.31956735, 0.07252306, 0.08111481, 0.81137309, 1.302895320, 2.12264151, 1.54109589, 8.86363636,
                     12.13692946, 0.40462028],
            'target': [0.06246232, 1.00000000, 0.34266478, 0.84750219, 14.80102041, 8.73795181, 1.52845100, 5.74455446,
                       0.285756500, 12.72368421, 0.6250808, 3.85771277, 6.95683453, 0.84923888, 0.40130032]
        },
        'cholect50-challenge': {
            'tool': [0.08495163, 0.88782288, 0.11259564, 2.61948830, 1.784866470, 1.144624170],
            'verb': [0.39862805, 0.06981640, 0.08332925, 0.81876204, 1.415868390, 2.269359150, 1.28428410, 7.35822511,
                     18.67857143, 0.45704490],
            'target': [0.07333818, 0.87139287, 0.42853950, 1.00000000, 17.67281106, 13.94545455, 1.44880997, 6.04889590,
                       0.326188650, 16.82017544, 0.63577586, 6.79964539, 6.19547658, 0.96284208, 0.51559559]
        },
        'cholect45-crossval': {
            1: {
                'tool': [0.08165644, 0.91226868, 0.10674758, 2.85418156, 1.60554885, 1.10640067],
                'verb': [0.37870137, 0.06836869, 0.07931255, 0.84780024, 1.21880342, 2.52836879, 1.30765704, 6.88888889,
                         17.07784431, 0.45241117],
                'target': [0.07149629, 1.0, 0.41013597, 0.90458015, 13.06299213, 12.06545455, 1.5213205, 5.04255319,
                           0.35808332, 45.45205479, 0.67493897, 7.04458599, 9.14049587, 0.97330595, 0.52633249]
            },
            2: {
                'tool': [0.0854156, 0.89535362, 0.10995253, 2.74936869, 1.78264429, 1.13234529],
                'verb': [0.36346863, 0.06771776, 0.07893261, 0.82842725, 1.33892161, 2.13049748, 1.26120359, 5.72674419,
                         19.7, 0.43189126],
                'target': [0.07530655, 0.97961957, 0.4325135, 0.99393438, 15.5387931, 14.5951417, 1.53862569,
                           6.01836394, 0.35184462, 15.81140351, 0.709506, 5.79581994, 8.08295964, 1.0, 0.52689272]
            },
            3: {
                "tool": [0.0915228, 0.89714969, 0.12057004, 2.72128174, 1.94092281, 1.12948557],
                "verb": [0.43636862, 0.07558554, 0.0891017, 0.81820519, 1.53645582, 2.31924198, 1.28565657, 6.49387755,
                         18.28735632, 0.48676763],
                "target": [0.06841828, 0.90980736, 0.38826607, 1.0, 14.3640553, 12.9875, 1.25939394, 5.38341969,
                           0.29060227, 13.67105263, 0.59168565, 6.58985201, 5.72977941, 0.86824513, 0.47682423]

            },
            4: {
                'tool': [0.08222218, 0.85414117, 0.10948695, 2.50868784, 1.63235867, 1.20593318],
                'verb': [0.41154261, 0.0692142, 0.08427214, 0.79895288, 1.33625219, 2.2624166, 1.35343681, 7.63,
                         17.84795322, 0.43970609],
                'target': [0.07536126, 0.85398445, 0.4085784, 0.95464422, 15.90497738, 18.5978836, 1.55875831,
                           5.52672956, 0.33700863, 15.41666667, 0.74755423, 5.4921875, 6.11304348, 1.0, 0.50641118],
            },
            5: {
                'tool': [0.0804654, 0.92271157, 0.10489631, 2.52302243, 1.60074906, 1.09141982],
                'verb': [0.50710436, 0.06590258, 0.07981184, 0.81538866, 1.29267277, 2.20525568, 1.29699248, 7.32311321,
                         25.45081967, 0.46733895],
                'target': [0.07119395, 0.87450495, 0.43043372, 0.86465981, 14.01984127, 23.7114094, 1.47577277,
                           5.81085526, 0.32129865, 22.79354839, 0.63304067, 6.92745098, 5.88833333, 1.0, 0.53175798]
            }
        },
        'cholect50-crossval': {
            1: {
                'tool': [0.0828851, 0.8876, 0.10830995, 2.93907285, 1.63884786, 1.14499484],
                'verb': [0.29628942, 0.07366916, 0.08267971, 0.83155428, 1.25402434, 2.38358209, 1.34938741, 7.56872038,
                         12.98373984, 0.41502079],
                'target': [0.06551745, 1.0, 0.36345711, 0.82434783, 13.06299213, 8.61818182, 1.4017744, 4.62116992,
                           0.32822238, 45.45205479, 0.67343211, 4.13200498, 8.23325062, 0.88527215, 0.43113306],

            },
            2: {
                'tool': [0.08586283, 0.87716737, 0.11068887, 2.84210526, 1.81016949, 1.16283571],
                'verb': [0.30072757, 0.07275414, 0.08350168, 0.80694143, 1.39209979, 2.22754491, 1.31448763, 6.38931298,
                         13.89211618, 0.39397505],
                'target': [0.07056703, 1.0, 0.39451115, 0.91977006, 15.86206897, 9.68421053, 1.44483706, 5.44378698,
                           0.31858714, 16.14035088, 0.7238395, 4.20571429, 7.98264642, 0.91360477, 0.43304307],
            },
            3: {
                'tool': [0.09225068, 0.87856006, 0.12195811, 2.82669323, 1.97710987, 1.1603972],
                'verb': [0.34285159, 0.08049804, 0.0928239, 0.80685714, 1.56125608, 2.23984772, 1.31471136, 7.08835341,
                         12.17241379, 0.43180428],
                'target': [0.06919395, 1.0, 0.37532866, 0.9830703, 15.78801843, 8.99212598, 1.27597765, 5.36990596,
                           0.29177312, 15.02631579, 0.64935557, 5.08308605, 5.86643836, 0.86580743, 0.41908257],
            },
            4: {
                'tool': [0.08247885, 0.83095539, 0.11050268, 2.58193042, 1.64497676, 1.25538881],
                'verb': [0.31890981, 0.07380354, 0.08804592, 0.79094077, 1.35928144, 2.17017208, 1.42947103, 8.34558824,
                         13.19767442, 0.40666428],
                'target': [0.07777646, 0.95894072, 0.41993829, 0.95592153, 17.85972851, 12.49050633, 1.65701092,
                           5.74526929, 0.33763901, 17.31140351, 0.83747083, 3.95490982, 6.57833333, 1.0, 0.47139615],
            },
            5: {
                'tool': [0.07891691, 0.89878025, 0.10267677, 2.53805556, 1.60636428, 1.12691169],
                'verb': [0.36420961, 0.06825313, 0.08060635, 0.80956984, 1.30757221, 2.09375, 1.33625848, 7.9009434,
                         14.1350211, 0.41429631],
                'target': [0.07300329, 0.97128713, 0.42084942, 0.8829883, 15.57142857, 19.42574257, 1.56521739,
                           5.86547085, 0.32732733, 25.31612903, 0.70171674, 4.55220418, 6.13125, 1.0, 0.48528321],
            }
        }
    }
    return switcher.get(case)


def assign_gpu(gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def fusion(predicted_list, labels):
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0 / len_layer for i in range(0, len_layer)]
    for out, w in zip(predicted_list, weight_list):
        resize_out = F.interpolate(out, size=labels.size(0), mode='nearest')
        resize_out_list.append(resize_out)
        if out.size(2) == labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            resize_label = F.interpolate(labels.float().transpose(0, 1).unsqueeze(0), size=out.size(2), mode='nearest')

            labels_list.append(resize_label.squeeze().transpose(0, 1).long())

        all_out_list.append(out)

    # sss
    return all_out, all_out_list, labels_list


def train_loop(dataloader, model, activation, loss_fn_ivt, optimizers, scheduler,
               epoch, writer, final_eval=False):
    start = time.time()
    for batch, (img, (y1, y2, y3, y4), paths) in enumerate(dataloader):
        if batch > len(dataloader) / FLAGS.train_div:
            break
        img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
        model.train()
        predicted_list, predicted_list_i, predicted_list_v, predicted_list_t, feature_list, prototype = model(img, True)

        # calculate loss
        loss_ivt, loss_i, loss_v, loss_t = [0, 0, 0, 0]
        logit_ivt = [p for p in predicted_list]
        logit_i = [p for p in predicted_list_i]
        logit_v = [p for p in predicted_list_v]
        logit_t = [p for p in predicted_list_t]

        _, resize_list, labels_list = fusion(logit_ivt, y4[0])
        for pd, la in zip(resize_list, labels_list):
            loss_ivt = loss_ivt + loss_fn_ivt(pd[0].transpose(0, 1), la.float())

        _, resize_list, labels_list = fusion(logit_i, y1[0])
        for pd, la in zip(resize_list, labels_list):
            loss_i = loss_i + loss_fn_i(pd[0].transpose(0, 1), la.float())

        _, resize_list, labels_list = fusion(logit_v, y2[0])
        for pd, la in zip(resize_list, labels_list):
            loss_v = loss_v + loss_fn_v(pd[0].transpose(0, 1), la.float())

        _, resize_list, labels_list = fusion(logit_t, y3[0])
        for pd, la in zip(resize_list, labels_list):
            loss_t = loss_t + loss_fn_t(pd[0].transpose(0, 1), la.float())

        if FLAGS.loss_type == 'i':
            loss = loss_i
        elif FLAGS.loss_type == 'v':
            loss = loss_v
        elif FLAGS.loss_type == 't':
            loss = loss_t
        elif FLAGS.loss_type == 'ivt':
            loss = loss_ivt
        elif FLAGS.loss_type == 'single':
            loss = (loss_i + loss_v + loss_t) / 3
        else:
            print(loss_i.item(), '-----', loss_v.item(), '-----', loss_t.item(), '-----', loss_ivt.item(), '-----',
                  epoch)
            loss = 0.1 * (loss_i + loss_v + loss_t) + loss_ivt

        total_step = (epoch) * len(dataloader) + batch + 1
        info_loss = {
            'loss_i': loss_i.item(),
            'loss_v': loss_v.item(),
            'loss_t': loss_t.item(),
            'loss_ivt': loss_ivt.item(),
            'loss': loss.item()
        }
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


def test_loop(dataloader, model, activation, writer, final_eval=False, mode='val'):
    global allstep, all_val_step
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mAP.reset()
    # final_eval = True
    # if final_eval and not set_chlg_eval:
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()

    with torch.no_grad():
        for batch, (img, (y1, y2, y3, y4), paths) in enumerate(dataloader):
            img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            model.eval()
            predicted_list, predicted_list_i, predicted_list_v, predicted_list_t, feature_list, prototype = model(img,
                                                                                                                  False)

            # calculate loss
            loss_ivt, loss_i, loss_v, loss_t = [0, 0, 0, 0]
            logit_ivt = [p for p in predicted_list]
            logit_i = [p for p in predicted_list_i]
            logit_v = [p for p in predicted_list_v]
            logit_t = [p for p in predicted_list_t]

            mAP.update(y4[0].float().detach().cpu(),
                       activation(logit_ivt[0][0].transpose(0, 1)).detach().cpu())  # Log metrics
            mAPi.update(y1[0].float().detach().cpu(),
                        activation(logit_i[0][0].transpose(0, 1)).detach().cpu())  # Log metrics
            mAPv.update(y2[0].float().detach().cpu(),
                        activation(logit_v[0][0].transpose(0, 1)).detach().cpu())  # Log metrics
            mAPt.update(y3[0].float().detach().cpu(),
                        activation(logit_t[0][0].transpose(0, 1)).detach().cpu())  # Log metrics

    mAP.video_end()
    mAPv.video_end()
    mAPt.video_end()
    mAPi.video_end()


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


import network_res

model_base = network_res.VideoNas(basename='resnet18').cuda()
# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
pytorch_total_params = sum(p.numel() for p in model_base.parameters())
pytorch_train_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
if FLAGS.pretrain_dir:
    model_base = load_model(model_base, FLAGS.pretrain_dir)
print('BackBone: Total params: %.2fM' % (sum(p.numel() for p in model_base.parameters()) / 1000000.0))
for params in model_base.parameters():
    params.requires_grad = False
model_base.eval()
# %% model
num_stages = 3  # refinement stages
num_layers = 12  # layers of prediction tcn e
num_f_maps = 512
dim = FLAGS.input_dim  # input dim
num_classes = 100
# print(args.num_classes)
num_layers_PG = FLAGS.num_layers_PG
num_layers_R = FLAGS.num_layers_R
num_R = FLAGS.num_R
model = network.VideoNas(FLAGS, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes).cuda()
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
# warmups[2] = int(epochs * 0.25)
optimizer_ivt = torch.optim.SGD(model.parameters(), lr=wp_lr[2], weight_decay=weight_decay)
scheduler_ivta = torch.optim.lr_scheduler.LinearLR(optimizer_ivt, start_factor=power, total_iters=warmups[2])
scheduler_ivtb = torch.optim.lr_scheduler.ExponentialLR(optimizer_ivt, gamma=decay_rate)
scheduler_ivt = torch.optim.lr_scheduler.SequentialLR(optimizer_ivt, schedulers=[scheduler_ivta, scheduler_ivtb],
                                                      milestones=[warmups[2] + 1])

lr_schedulers = [scheduler_ivt]
optimizers = [optimizer_ivt]

print("Model's weight loaded ...")
# (32, 3, 256, 448)
# (32, 6)
# (32, 10)
# (32, 15)
# (32, 100)
# %% data loading : variant and split selection (Note: original paper used different augumentation per epoch)

dataset = dataloader.CholecT50(
    args=FLAGS,
    dataset_dir=data_dir,
    dataset_variant=dataset_variant,
    test_fold=kfold,
    augmentation_list=data_augmentations,
    model=model_base
)

# build dataset
train_dataset, val_dataset, test_dataset, test_train_dataset = dataset.build()

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, prefetch_factor=10 * 1,
                              num_workers=10, pin_memory=True, persistent_workers=True, drop_last=False)
val_dataloaders = []
for video_dataset in val_dataset:
    val_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                prefetch_factor=5 * 1,
                                num_workers=5, pin_memory=True, persistent_workers=True, drop_last=False)
    val_dataloaders.append(val_dataloader)
# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                 prefetch_factor=5 * 1,
                                 num_workers=5, pin_memory=True, persistent_workers=True, drop_last=False)
    test_dataloaders.append(test_dataloader)
test_train_dataloaders = []
for video_dataset in test_train_dataset:
    test_train_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                       prefetch_factor=5 * 1,
                                       num_workers=5, pin_memory=True, persistent_workers=True, drop_last=False)
    test_train_dataloaders.append(test_train_dataloader)
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
            # Train
            print("Traning | lr: {} | epoch {}".format([op.state_dict()['param_groups'][0]['lr'] for op in optimizers],
                                                       epoch), end=" | ",
                  file=open(logfile, 'a+'))
            train_loop(train_dataloader, model, activation, loss_fn_ivt, optimizers,
                       lr_schedulers, epoch, writer)

            # val
            if epoch % val_interval == 0:
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
                elif FLAGS.loss_type == 'single':
                    mean_mAP = (mAPi.compute_video_AP()['mAP'] + mAPv.compute_video_AP()['mAP'] +
                                mAPt.compute_video_AP()['mAP']) / 3
                    behaviour = weight_mgt(mean_mAP, epoch=epoch)
                else:
                    behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                if FLAGS.loss_type in ['i', 'v', 't', 'single']:
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
                                                                                                 mAP.compute_video_AP(
                                                                                                     'ivt',
                                                                                                     ignore_null=set_chlg_eval)[
                                                                                                     'mAP']),
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
                if FLAGS.loss_type in ['i', 'v', 't', 'single']:
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
                                                                                                 mAP.compute_video_AP(
                                                                                                     'ivt',
                                                                                                     ignore_null=set_chlg_eval)[
                                                                                                     'mAP']),
                    file=open(logfile, 'a+'))
        except KeyboardInterrupt:
            print(f'>> Process cancelled by user at {time.ctime()}, ...', file=open(logfile, 'a+'))
            sys.exit(1)
    test_ckpt = ckpt_path

# %% eval
if is_test:
    print("Test weight: ", test_ckpt)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if is_train:
        tags = ['latest', 'best']
    else:
        tags = ['best']
    for tag in tags:
        print('========', tag, '==============')
        if tag == 'best':
            model.load_state_dict(torch.load(test_ckpt))

        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        for test_dataloader in test_dataloaders:
            test_loop(test_dataloader, model, activation, writer, final_eval=True, mode='test')

        mAPs = {'ivt': mAP, 'i': mAPi, 'v': mAPv, 't': mAPt}
        import pickle

        f = open(model_dir + '/mAPs_' + tag + '_k' + str(kfold) + '.pckl', 'wb')
        pickle.dump(mAPs, f)
        f.close()
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
