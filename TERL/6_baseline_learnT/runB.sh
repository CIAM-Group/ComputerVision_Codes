#!/usr/bin/env bash
GPU=0
K=3
PROTO=1
CON=1
BB='swin_B_384_22k'
cd ..
cd 6_baseline_learnT
SEED=20000912
VERSION='240302_6_baseline_learnT_swinB_div4_p'${PROTO}'_con'${CON}'_k'${K}'_seed'${SEED}
DIM=1024
python run.py -t -e --seed ${SEED} --loss_type all --moco-dim ${DIM} --moco-k 16384 --mlp --opt_type sgd --w_con ${CON} --w_proto ${PROTO} --w_epoch 1 --drop_rate 0.0 --train_div 4 --power 0.1 --weight_decay 1e-5 --dataset_variant=cholect45-crossval --kfold $K --epochs=20 --batch=16 -l 0.001 0.001 1e-5 --img_size 384 --backbone ${BB} --hidden_dim ${DIM} --version=${VERSION} --gpu ${GPU} --val_interval 2
python test.py -e --seed ${SEED} --loss_type all --moco-dim ${DIM} --mlp --moco-k 16384 --img_size 384 --backbone ${BB} --hidden_dim ${DIM} --dataset_variant=cholect45-crossval --kfold $K --epochs=200 --batch=16 -l 0.001 0.001 1e-5 --version1 ${VERSION} --version=test_${VERSION} --gpu ${GPU} --val_interval 5

cd ..
cd 0_5fold_TCN_black
python run.py -t -e --seed ${SEED} --input_dim ${DIM} --mask --pos_w 'pre_w' --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=$K --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${VERSION}_TCN_${SEED} --version1=${VERSION} --gpu ${GPU} --val_interval 20
