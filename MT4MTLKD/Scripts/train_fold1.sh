#!/usr/bin/env bash
GPU='1'
KFOLD=1

# Teacher training
VERSION='SwinL'
TASK='i'
IN_DIM=1536
IM_SIZE=384
cd ..
cd Spatial_transformer
python run.py -t -e --img_size ${IM_SIZE} --backbone swin_L_${IM_SIZE}_22k --hidden_dim ${IN_DIM} --loss_type ${TASK} --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=100 --batch=16 -l 1e-2 5e-3 1e-5 --version=${VERSION} --gpu ${GPU} --val_interval 5
python test.py -e --img_size ${IM_SIZE} --backbone swin_L_${IM_SIZE}_22k --hidden_dim ${IN_DIM} --loss_type ${TASK} --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=100 --batch=16 -l 1e-2 5e-3 1e-5 --version=${VERSION} --gpu ${GPU} --val_interval 5
cd ..
cd Temporal_mstct
python run.py -t -e --loss_type ${TASK} --input_dim ${IN_DIM} --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=2000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 500 --decay_rate 0.999 --version=${VERSION}_MSTCT --version1=${VERSION} --gpu ${GPU} --val_interval 20
python test.py -e --loss_type ${TASK} --input_dim ${IN_DIM} --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=2000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 500 --decay_rate 0.999 --version=${VERSION}_MSTCT --version1=${VERSION} --gpu ${GPU} --val_interval 20

# Student training

VERSION_S='SwinL2Res18'
cd ..
cd Spatial_cnn
python run.py -t -e --rates 1 1 1 --temp 4 --network resnet18 --teacher_feat_version ${VERSION} --teacher_pred_version ${VERSION}_MSTCT --student_dim 512 --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${VERSION_S} --gpu ${GPU} --val_interval 10
python test.py -e --rates 1 1 1 --temp 4 --network resnet18 --student_dim 512 --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${VERSION_S} --gpu ${GPU} --val_interval 10
cd ..
cd Temporal_tenco
python run.py -t -e --seed 19991111 --mask --input_dim 512 --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${VERSION_S}_TCN --version1=${VERSION_S} --gpu ${GPU} --val_interval 20
