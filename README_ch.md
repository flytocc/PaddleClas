# Rethinking Bottleneck Structure for Efficient Mobile Network Design

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

这是一个PaddlePaddle实现的 MobileNeXt 。

**论文:**
[Rethinking Bottleneck Structure for Efficient Mobile Network Design](https://arxiv.org/pdf/2007.02269.pdf)

**参考repo:**
[MobileNeXt](https://github.com/yitu-opensource/MobileNeXt) &
[rethinking_bottleneck_design](https://github.com/zhoudaquan/rethinking_bottleneck_design)

在此非常感谢`zhoudaquan`和`yitutech-opensource`等人的贡献，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

### 2.1 数据集

[ImageNet](https://image-net.org/)项目是一个大型视觉数据库，用于视觉目标识别研究任务，该项目已手动标注了 1400 多万张图像。ImageNet-1k 是 ImageNet 数据集的子集，其包含 1000 个类别。训练集包含 1281167 个图像数据，验证集包含 50000 个图像数据。2010 年以来，ImageNet 项目每年举办一次图像分类竞赛，即 ImageNet 大规模视觉识别挑战赛（ILSVRC）。挑战赛使用的数据集即为 ImageNet-1k。到目前为止，ImageNet-1k 已经成为计算机视觉领域发展的最重要的数据集之一，其促进了整个计算机视觉的发展，很多计算机视觉下游任务的初始化模型都是基于该数据集训练得到的。

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

### 2.2 复现精度

| 模型            | epochs | top1 acc (参考精度) | (复现精度) | 权重                 \| 训练日志   |
|:--------------:|:------:|:------------------:|:---------:|:--------------------------------:|
| MobileNeXt-1.0 |  200   | 74.022             | 74.024    | best_model.pdparams \| train.log |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1Kt5Bk6PhlrCSs4Ie5hwamg?pwd=cp32)


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
git checkout -b mobilenext
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

参考 [2.1 数据集](#21-数据集)，从官方下载数据后，按如下格式组织数据，即可在 PaddleClas 中使用 ImageNet1k 数据集进行训练。

```bash
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

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml
```

cooldown

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100_cooldown.yaml
```

部分训练日志如下所示。

```
[2022/09/01 13:08:54] ppcls INFO: [Train][Epoch 187/200][Iter: 1350/2502]lr(LinearWarmup): 0.00125084, CELoss: 2.16119, loss: 2.16119, batch_cost: 0.53062s, reader_cost: 0.11223, ips: 241.22737 samples/s, eta: 4:57:50
[2022/09/01 13:09:19] ppcls INFO: [Train][Epoch 187/200][Iter: 1400/2502]lr(LinearWarmup): 0.00125084, CELoss: 2.16252, loss: 2.16252, batch_cost: 0.53053s, reader_cost: 0.62890, ips: 241.26978 samples/s, eta: 4:57:20
```

### 4.2 模型评估

``` shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/eval.py \
    -c ./ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml \
    -o Global.pretrained_model=$TRAINED_MODEL
```

### 4.3 模型预测

```shell
python tools/infer.py \
    -c ./ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml \
    -o Infer.infer_imgs=./deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg \
    -o Global.pretrained_model=$TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 246, 236, 209], 'scores': [0.85077, 0.0157, 0.01342, 0.00362, 0.00354], 'file_name': './deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'Chesapeake Bay retriever']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.85077`。

### 4.4 模型导出

```shell
python tools/export_model.py \
    -c ./ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml \
    -o Global.save_inference_dir=./deploy/models/class_MobileNeXt_100_ImageNet_infer \
    -o Global.pretrained_model=$TRAINED_MODEL

python deploy/python/predict_cls.py \
    -c deploy/configs/inference_cls.yaml \
    -o Global.cpu_num_threads=1 \
    -o Global.infer_imgs=./deploy/images/ImageNet/ILSVRC2012_val_00020010.jpeg \
    -o Global.inference_model_dir=./deploy/models/class_MobileNeXt_100_ImageNet_infer \
    -o PreProcess.transform_ops.0.ResizeImage.interpolation=bicubic \
    -o PreProcess.transform_ops.0.ResizeImage.backend=pil \
    -o PostProcess.Topk.class_id_map_file=./ppcls/utils/imagenet1k_label_list.txt
```

输出结果为
```
ILSVRC2012_val_00020010.jpeg:   class id(s): [178, 211, 246, 236, 209], score(s): [0.85, 0.02, 0.01, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'Chesapeake Bay retriever']
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.85`。与predict.py结果的误差在正常范围内。


## 5. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [TIPC: test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-*-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/configs/MobileNeXt/MobileNeXt_100_train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/MobileNeXt/MobileNeXt_100_train_infer_python.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
Run successfully with command - MobileNeXt_100 - python3 tools/train.py -c ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml -o Global.seed=1234 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.loader.num_workers=0 -o DataLoader.Train.loader.use_shared_memory=False -o Global.device=gpu  -o Global.output_dir=./test_tipc/output/MobileNeXt_100/lite_train_lite_infer/norm_train_gpus_0_autocast_null -o Global.epochs=2     -o DataLoader.Train.sampler.batch_size=8   !
Run successfully with command - MobileNeXt_100 - python3 tools/eval.py -c ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml -o Global.pretrained_model=./test_tipc/output/MobileNeXt_100/lite_train_lite_infer/norm_train_gpus_0_autocast_null/MobileNeXt_100/latest -o Global.device=gpu  !
Run successfully with command - MobileNeXt_100 - python3 tools/export_model.py -c ppcls/configs/ImageNet/MobileNeXt/MobileNeXt_100.yaml -o Global.pretrained_model=./test_tipc/output/MobileNeXt_100/lite_train_lite_infer/norm_train_gpus_0_autocast_null/MobileNeXt_100/latest -o Global.save_inference_dir=./test_tipc/output/MobileNeXt_100/lite_train_lite_infer/norm_train_gpus_0_autocast_null!
Run successfully with command - MobileNeXt_100 - python3 python/predict_cls.py -c configs/inference_cls.yaml -o PreProcess.transform_ops.0.ResizeImage.interpolation=bicubic -o PreProcess.transform_ops.0.ResizeImage.backend=pil -o Global.use_gpu=True -o Global.use_tensorrt=False -o Global.use_fp16=False -o Global.inference_model_dir=.././test_tipc/output/MobileNeXt_100/lite_train_lite_infer/norm_train_gpus_0_autocast_null -o Global.batch_size=1 -o Global.infer_imgs=../dataset/ILSVRC2012/val -o Global.benchmark=True > .././test_tipc/output/MobileNeXt_100/lite_train_lite_infer/infer_gpu_usetrt_False_precision_False_batchsize_1.log 2>&1 !
......
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。


## 6. License

This project is released under BSD License.


## 7. 参考链接与文献

1. Rethinking Bottleneck Structure for Efficient Mobile Network Design: https://arxiv.org/pdf/2007.02269.pdf
2. MobileNeXt: https://github.com/yitu-opensource/MobileNeXt
3. rethinking_bottleneck_design: https://github.com/zhoudaquan/rethinking_bottleneck_design

```
@article{zhou2020rethinking,
  title={Rethinking Bottleneck Structure for Efficient Mobile Network Design},
  author={Zhou, Daquan and Hou, Qibin and Chen, Yunpeng and Feng, Jiashi and Yan, Shuicheng},
  journal={ECCV, August},
  year={2020}
}
```
