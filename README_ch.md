# CvT: Introducing Convolutions to Vision Transformers

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
   * [2.1 数据集](#21-数据集)
   * [2.2 复现精度](#22-复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. 自动化测试脚本](#5-自动化测试脚本)
* [6. License](#6-license)
* [7. 参考链接与文献](#7-参考链接与文献)


## 1. 简介

这是一个PaddlePaddle实现的 CvT 。

![](https://github.com/microsoft/CvT/blob/main/figures/pipeline.svg)

**论文:**
[CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808)

**参考repo:**
[CvT](https://github.com/microsoft/CvT)

在此非常感谢`awindsor`和`lmk123568`等人的贡献，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

### 2.1 数据集

[ImageNet](https://image-net.org/)项目是一个大型视觉数据库，用于视觉目标识别研究任务，该项目已手动标注了 1400 多万张图像。ImageNet-1k 是 ImageNet 数据集的子集，其包含 1000 个类别。训练集包含 1281167 个图像数据，验证集包含 50000 个图像数据。2010 年以来，ImageNet 项目每年举办一次图像分类竞赛，即 ImageNet 大规模视觉识别挑战赛（ILSVRC）。挑战赛使用的数据集即为 ImageNet-1k。到目前为止，ImageNet-1k 已经成为计算机视觉领域发展的最重要的数据集之一，其促进了整个计算机视觉的发展，很多计算机视觉下游任务的初始化模型都是基于该数据集训练得到的。

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

### 2.2 复现精度

| 模型            | epochs | top1 acc (参考精度) | (复现精度) | 权重                 \| 训练日志   |
|:--------------:|:------:|:------------------:|:---------:|:--------------------------------:|
| cvt_13_224x224 |  300   | 81.6               | 81.6      | best_model.pdparams \| train.log |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1dhrv6DBb-LC_z3sv53ZobQ?pwd=uqch)


## 3. 准备数据与环境

### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * 3090
- 框架：
  - PaddlePaddle == 2.3.1
  - Pillow == 8.4.0

* 安装paddlepaddle

```bash
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.3.1
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/PaddleClas.git
cd PaddleClas
git checkout -b CvT
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

参考 [2.1 数据集](#21-数据集)，从官方下载数据后，按如下格式组织数据，即可在 PaddleClas 中使用 ImageNet1k 数据集进行训练。

```
imagenet/
    |_ train/
    |  |_ n01440764
    |  |  |_ n01440764_10026.JPEG
    |  |  |_ ...
    |  |_ ...
    |  |
    |  |_ n15075141
    |     |_ ...
    |     |_ n15075141_9993.JPEG
    |_ val/
    |  |_ ILSVRC2012_val_00000001.JPEG
    |  |_ ...
    |  |_ ILSVRC2012_val_00050000.JPEG
    |_ train_list.txt
    |_ val_list.txt
```


## 4. 开始使用

### 4.1 模型训练

* 单机多卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml \
    -o Global.update_freq=4
```

部分训练日志如下所示。

```
[2022/09/15 11:55:32] ppcls INFO: [Train][Epoch 260/300][Iter: 1000/2500]lr(LinearWarmup): 0.00010335, CELoss: 2.84290, loss: 2.84290, batch_cost: 0.60855s, reader_cost: 0.01440, ips: 210.33667 samples/s, eta: 17:10:17
[2022/09/15 11:56:02] ppcls INFO: [Train][Epoch 260/300][Iter: 1050/2500]lr(LinearWarmup): 0.00010335, CELoss: 2.84662, loss: 2.84662, batch_cost: 0.60854s, reader_cost: 0.01392, ips: 210.33875 samples/s, eta: 17:09:46
```

### 4.2 模型评估

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/eval.py \
    -c ./ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml \
    -o Global.pretrained_model=$TRAINED_MODEL
```

### 4.3 模型预测

```bash
python tools/infer.py \
    -c ./ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml \
    -o Infer.infer_imgs=./deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg \
    -o Global.pretrained_model=$TRAINED_MODEL
```

<div align="center">
    <img src="./deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 210, 246, 268], 'scores': [0.83223, 0.00164, 0.00102, 0.0009, 0.00086], 'file_name': './deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'German short-haired pointer', 'Great Dane', 'Mexican hairless']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.83223`。

### 4.4 模型导出

```bash
python tools/export_model.py \
    -c ./ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml \
    -o Global.save_inference_dir=./deploy/models/class_cvt_13_224x224_ImageNet_infer \
    -o Global.pretrained_model=$TRAINED_MODEL

python deploy/python/predict_cls.py \
    -c deploy/configs/inference_cls.yaml \
    -o Global.cpu_num_threads=1 \
    -o Global.infer_imgs=./deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg \
    -o Global.inference_model_dir=./deploy/models/class_cvt_13_224x224_ImageNet_infer \
    -o PreProcess.transform_ops.0.ResizeImage.interpolation=bicubic \
    -o PreProcess.transform_ops.0.ResizeImage.backend=pil \
    -o PostProcess.Topk.class_id_map_file=./ppcls/utils/imagenet1k_label_list.txt
```

输出结果为
```
ILSVRC2012_val_00020010.jpeg:   class id(s): [178, 211, 210, 246, 268], score(s): [0.83, 0.00, 0.00, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'German short-haired pointer', 'Great Dane', 'Mexican hairless']
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.83`。与predict.py结果的误差在正常范围内。


## 5. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [TIPC: test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```bash
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-*-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/configs/CvT/cvt_13_224x224_train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/CvT/cvt_13_224x224_train_infer_python.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
successfully with command - cvt_13_224x224 - python3 tools/train.py -c ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml -o Global.seed=1234 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.loader.num_workers=0 -o DataLoader.Train.loader.use_shared_memory=False -o Global.device=gpu -o Global.output_dir=/home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1 -o Global.epochs=2     -o DataLoader.Train.sampler.batch_size=8    - /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1.log !
successfully with command - cvt_13_224x224 - python3 tools/eval.py -c ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml -o Global.pretrained_model=/home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1/cvt_13_224x224/latest -o Global.device=gpu   > /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_eval.log 2>&1 - /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_eval.log !
successfully with command - cvt_13_224x224 - python3 tools/export_model.py -c ppcls/configs/ImageNet/CvT/cvt_13_224x224.yaml -o Global.pretrained_model=/home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1/cvt_13_224x224/latest -o Global.save_inference_dir=/home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1 > /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_export.log 2>&1 - /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_export.log !
successfully with command - cvt_13_224x224 - python3 python/predict_cls.py -c configs/inference_cls.yaml -o PreProcess.transform_ops.0.ResizeImage.interpolation=bicubic -o PreProcess.transform_ops.0.ResizeImage.backend=pil -o Global.use_gpu=True -o Global.use_tensorrt=False -o Global.use_fp16=False -o Global.inference_model_dir=/home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1 -o Global.batch_size=1 -o Global.infer_imgs=../dataset/ILSVRC2012/val/ILSVRC2012_val_00000001.JPEG -o Global.benchmark=False > /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/python_infer_gpu_gpus_0_usetrt_False_precision_False_batchsize_1.log 2>&1  - /home/nieyang/PaddleClas/test_tipc/output/cvt_13_224x224/lite_train_lite_infer/python_infer_gpu_gpus_0_usetrt_False_precision_False_batchsize_1.log !
......
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。


## 6. License

This project is released under MIT License.

If you find this work or code is helpful in your research, please cite:
```
@article{wu2021cvt,
  title={Cvt: Introducing convolutions to vision transformers},
  author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
  journal={arXiv preprint arXiv:2103.15808},
  year={2021}
}
```

## 7. 参考链接与文献

1. CvT: Introducing Convolutions to Vision Transformers: https://arxiv.org/abs/2103.15808
2. CvT: https://github.com/microsoft/CvT
