#!/usr/bin/env bash
GPU='0'
KFOLD=1
VERSION='SwinL2Res18'
cd ..
cd Spatial_cnn
python test.py -e --rates 1 1 1 --temp 4 --soft_type KL_T --network resnet18 --student_dim 512 --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${VERSION} --gpu ${GPU} --val_interval 10
cd ..
cd Temporal_tenco
python run.py -e --seed 19991111 --mask --input_dim 512 --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${VERSION}_TCN --version1=${VERSION} --gpu ${GPU} --val_interval 20 --test_ckpt ./__checkpoint__/run_${VERSION}_TCN/rendezvous_l8_cholectcholect45-crossval_k${KFOLD}_batchnorm_lowres_latest.pth
