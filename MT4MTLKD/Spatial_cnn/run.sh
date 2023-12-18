#!/usr/bin/env bash
GPU='0'
KFOLD=1
epoch=1000
seed=20230906
TIME='231020'
python run.py -t -e --rates 1 1 1 --temp 4 --soft_type KL_T --core_type rand --network resnet18 --student_dim 512 --spatialKD --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${TIME}_SwinL_to_Res18_rand --gpu ${GPU} --val_interval 10
python run.py -t -e --rates 1 1 1 --temp 4 --soft_type KL_T --core_type rand --network resnet18 --student_dim 512 --spatialKD --loss_type all --dataset_variant=cholect45-crossval --kfold 1 --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=KD_spatial_all_feat_ra111_KLT --gpu 0 --val_interval 10
cd ..
cd 2_5fold_KD_spatial_feat_savefeats
python test.py -e --rates 1 1 1 --temp 4 --soft_type KL_T --network resnet18 --student_dim 512 --spatialKD --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${TIME}_SwinL_to_Res18_rand --gpu ${GPU} --val_interval 10
cd ..
cd 2_5fold_KD_temp_nocausal
python run.py -t -e --seed ${seed} --mask --input_dim 512 --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=${epoch} --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${TIME}_SwinL_to_Res18_rand_${seed} --version1=${TIME}_SwinL_to_Res18_rand --gpu ${GPU} --val_interval 20 --pretrain_dir ../0-5fold/__checkpoint__/run_Benchmark_surgical_Resnet18_all_k1_lr0005/rendezvous_l8_cholectcholect45-crossval_k1_batchnorm_lowres.pth