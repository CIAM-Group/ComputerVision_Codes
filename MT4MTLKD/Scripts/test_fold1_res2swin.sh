#!/usr/bin/env bash
GPU='0'
KFOLD=1
VERSION='Res182SwinL'
cd ..
cd Spatial_cnn
python test.py -e --rates 1 1 1 --temp 4 --soft_type KL_T --img_size 384 --backbone swin_L_384_22k --hidden_dim 1536 --spatialKD --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=100 --batch=8 -l 1e-2 5e-3 1e-3 -w 9 18 38 --version=${VERSION} --gpu ${GPU} --val_interval 5
cd ..
cd Temporal_tenco
python run.py -e --seed 19991111 --mask --input_dim 512 --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${VERSION}_TCN --version1=${VERSION} --gpu ${GPU} --val_interval 20 --test_ckpt ./__checkpoint__/run_${VERSION}_TCN/rendezvous_l8_cholectcholect45-crossval_k${KFOLD}_batchnorm_lowres_latest.pth
