# Copyright (c) 2023 CIAM Group
**The code can only be used for non-commercial purposes. Please contact the authors if you want to use this code for business matters.**  

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# **MT4MTL-KD**: A Multi-teacher Knowledge Distillation Framework for Triplet Recognition

<i>Shuangchun Gui, Zhenkun Wang, Jixiang Chen, Xun Zhou, Chen Zhang, and Yi Cao</i>

<img src="imgs/3_metd_framework.jpg" width="100%">

This repository contains the pytorch implementation code and evaluation scripts. <br />


## Installation
```
conda env create -f environment.yaml
```
The code has been test on Linux operating system. It runs on GPU with Python 3.8.

<br />

## Data Preparation
Download [CholecT45 dataset](https://forms.gle/jTdPJnZCmSe2Daw7A)

<details>
  <summary>  
 Expand this to visualize the dataset directory structure.
  </summary>
  
  ```
    ──CholecT45
        ├───data
        │   ├───VID01
        │   │   ├───000000.png
        │   │   ├───000001.png
        │   │   ├───000002.png
        │   │   ├───
        │   │   └───N.png
        │   ├───VID02
        │   │   ├───000000.png
        │   │   ├───000001.png
        │   │   ├───000002.png
        │   │   ├───
        │   │   └───N.png
        │   ├───
        │   ├───
        │   ├───
        │   |
        │   └───VIDN
        │       ├───000000.png
        │       ├───000001.png
        │       ├───000002.png
        │       ├───
        │       └───N.png
        |
        ├───triplet
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───instrument
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───verb
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───target
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───dict
        │   ├───triplet.txt
        │   ├───instrument.txt
        │   ├───verb.txt
        │   ├───target.txt
        │   └───maps.txt
        |
        ├───LICENSE
        └───README.md
   ```
</details>

<br>

## Evaluation
* Download [our models](https://drive.google.com/file/d/1htWfsopwfHx5VGTcBx35r_yL_OJhEZYi/view?usp=drive_link)
* Put them under `./Spatial_cnn/__checkpoint__/run_SwinL2Res18` and `./Temporal_tenco/__checkpoint__/run_SwinL2Res18_TCN`
* cd `Scripts`
* Run scripts in `test_fold1.sh` to start the evaluation process
* Fold 1 results of the CholecT45 cross-validation split (Table V in the paper):

||Components AP ||||| Association AP |||
:---:|:---:|:---:|:---: |:---:|:---:|:---:|:---:|:---:|
AP<sub>I</sub> | AP<sub>V</sub> | AP<sub>T</sub> ||| AP<sub>IV</sub> | AP<sub>IT</sub> | AP<sub>IVT</sub> |
89.87 | 70.60 | 50.20 ||| 41.84 | 44.25 | 35.88 |

<br />

## Training
More details are in `./Scripts/train_fold1.sh`
* teacher model training: spatial model

```
cd Spatial_transformer
python run.py -t -e --img_size ${IM_SIZE} --backbone swin_L_${IM_SIZE}_22k --hidden_dim ${IN_DIM} --loss_type ${TASK} --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=100 --batch=16 -l 1e-2 5e-3 1e-5 --version=${VERSION} --gpu ${GPU} --val_interval 5
python test.py -e --img_size ${IM_SIZE} --backbone swin_L_${IM_SIZE}_22k --hidden_dim ${IN_DIM} --loss_type ${TASK} --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=100 --batch=16 -l 1e-2 5e-3 1e-5 --version=${VERSION} --gpu ${GPU} --val_interval 5
```

* teacher model training: temporal model

```
cd Temporal_mstct
python run.py -t -e --loss_type ${TASK} --input_dim ${IN_DIM} --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=2000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 500 --decay_rate 0.999 --version=${VERSION}_MSTCT --version1=${VERSION} --gpu ${GPU} --val_interval 20
python test.py -e --loss_type ${TASK} --input_dim ${IN_DIM} --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=2000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 500 --decay_rate 0.999 --version=${VERSION}_MSTCT --version1=${VERSION} --gpu ${GPU} --val_interval 20
```

* student model training: spatial model

```
cd Spatial_cnn
python run.py -t -e --rates 1 1 1 --temp 4 --network resnet18 --teacher_feat_version ${VERSION} --teacher_pred_version ${VERSION}_MSTCT --student_dim 512 --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${VERSION_S} --gpu ${GPU} --val_interval 10
python test.py -e --rates 1 1 1 --temp 4 --network resnet18 --student_dim 512 --loss_type all --dataset_variant=cholect45-crossval --kfold ${KFOLD} --epochs=200 --batch=8 -l 1e-2 5e-3 1e-3 --version=${VERSION_S} --gpu ${GPU} --val_interval 10
```

* student model training: temporal model

```
cd Temporal_tenco
python run.py -t -e --seed 19991111 --mask --input_dim 512 --loss_type all --fpn --dataset_variant=cholect45-crossval --kfold=${KFOLD} --epochs=1000 --batch=31 -l 1e-2 5e-3 1e-2 -w 9 18 200 --version=${VERSION_S}_TCN --version1=${VERSION_S} --gpu ${GPU} --val_interval 20
```

## Acknowledgements
MT4MTL-KD's implementation is based on the code of [RDV](https://github.com/CAMMA-public/rendezvous), [Q2L](https://github.com/SlongLiu/query2labels), [MS-TCT](https://github.com/dairui01/MS-TCT), and [SAHC](https://github.com/xmed-lab/SAHC). Thanks to them.


## Citation

If this code is useful for your research, please consider citing:

  ```shell
@article{gui2023mt4mtl,
  title={MT4MTL-KD: A Multi-teacher Knowledge Distillation Framework for Triplet Recognition},
  author={Gui, Shuangchun and Wang, Zhenkun and Chen, Jixiang and Zhou, Xun and Zhang, Chen and Cao, Yi},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}

  ```

## Note
* Contact: Shuangchun Gui (12132667@mail.sustech.edu.cn)
