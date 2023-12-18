#!/usr/bin/env bash
GPU='0'
KFOLD=1
VERSION='KD_spatial_all_feat_ra111_KLT'
python run.py -t -e --seed 19991111 --mask --input_dim 512 --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${VERSION}_TCN --version1=${VERSION} --gpu ${GPU} --val_interval 20
