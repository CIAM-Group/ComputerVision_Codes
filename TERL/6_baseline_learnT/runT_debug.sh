#!/usr/bin/env bash

python run.py -t -e --seed 20000912 --loss_type all --moco-dim 768 --moco-k 16384 --mlp --opt_type sgd --w_con 1 --w_proto 1 --w_epoch 1 --drop_rate 0.0 --train_div 2 --power 0.1 --weight_decay 1e-5 --dataset_variant=cholect45-crossval --kfold 1 --epochs=20 --batch=16 -l 0.001 0.001 1e-5 --img_size 224 --backbone swin_T_224_1k --hidden_dim 768 --version=240304_6_baseline_learnT_swinT_div2_p1_con1_k1_seed20000912 --gpu 0 --val_interval 2
python test.py -e --seed 20000912 --loss_type all --moco-dim 768 --mlp --moco-k 16384 --img_size 224 --backbone swin_T_224_1k --hidden_dim 768 --dataset_variant=cholect45-crossval --kfold 1 --epochs=200 --batch=16 -l 0.001 0.001 1e-5 --version1 240304_6_baseline_learnT_swinT_div2_p1_con1_k1_seed20000912 --version=test_240304_6_baseline_learnT_swinT_div2_p1_con1_k1_seed20000912 --gpu 0 --val_interval 5

cd ..
cd 0_5fold_TCN_black
python run.py -t -e --seed 20000912 --input_dim 768 --mask --pos_w pre_w --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=1 --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=240304_6_baseline_learnT_swinT_div2_p1_con1_k1_seed20000912_TCN_20000912 --version1=240304_6_baseline_learnT_swinT_div2_p1_con1_k1_seed20000912 --gpu 0 --val_interval 20