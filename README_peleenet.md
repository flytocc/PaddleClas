# Pelee: A Real-Time Object Detection System on Mobile Devices

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
   * [3.3 准备模型](#33-准备模型)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
* [5. 模型推理部署](#5-模型推理部署)
* [6. 自动化测试脚本](#6-自动化测试脚本)
* [7. LICENSE](#7-license)
* [8. 参考链接与文献](#8-参考链接与文献)


## 1. 简介

这是一个PaddlePaddle实现的 peelenet backbone

**论文:** [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/pdf/1804.06882.pdf)

**参考repo:** [PeleeNet](https://github.com/Robert-JunWang/PeleeNet)


在此非常感谢`Robert-JunWang`等人贡献的[PeleeNet](https://github.com/Robert-JunWang/PeleeNet)，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

aistudio[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/79807)

| 模型           |  epochs   |  acc   | 权重 | 日志    |
|----------|-----|-----|--|----|
| PeleeNet-论文  | 120+20   |   72.6  | - |   -  |
| PeleeNet-复现  |  200   |  71.9   |[link](./ckpt_and_log/200_epoch.pdparams) |  [link](./ckpt_and_log/200_epoch.log)   |
| PeleeNet-复现  |  300 |   72.2  | [link](./ckpt_and_log/300_epoch.pdparams) |   [link](./ckpt_and_log/300_epoch.log)  |


## 3. 准备数据与环境


### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * v100
- 框架：
  - PaddlePaddle >= 2.2.0

* 下载代码

```bash
git clone https://github.com/renmada/PaddleClas-peleenet
cd PaddleClas
```

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

aistudio[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/79807)


### 3.3 准备模型


如果您希望直接体验评估或者预测推理过程，可以直接根据第2章的内容下载提供的预训练模型，直接体验模型评估、预测、推理部署等内容。


## 4. 开始使用


### 4.1 模型训练

* 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/train.py \
    -c ./ppcls/configs/ImageNet/PeleeNet/PeleeNet.yaml
```

部分训练日志如下所示。

```
2022/04/30 11:28:57] ppcls INFO: [Train][Epoch 1/300][Iter: 0/1252]lr: 0.00004, CELoss: 7.01128, loss: 7.01128, batch_cost: 8.10043s, reader_cost: 7.31687, ips: 31.60325 images/sec, eta: 35 days, 5:08:42
[2022/04/30 11:30:30] ppcls INFO: [Train][Epoch 1/300][Iter: 100/1252]lr: 0.00403, CELoss: 6.97696, loss: 6.97696, batch_cost: 0.94871s, reader_cost: 0.22499, ips: 269.84145 images/sec, eta: 4 days, 2:57:18
```

### 4.2 模型评估

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAINED_MODEL=ckpt_and_log/300epoch
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/eval.py \
    -c ./ppcls/configs/ImageNet/PeleeNet/PeleeNet.yaml \
    -o Global.pretrained_model=$TRAINED_MODEL
```

### 4.3 模型预测

```shell
export TRAINED_MODEL=ckpt_and_log/300epoch
python tools/infer.py \
    -c ./ppcls/configs/ImageNet/PeleeNet/PeleeNet.yaml \
    -o Infer.infer_imgs=./deploy/images/ILSVRC2012_val_00020010.jpeg \
    -o Global.pretrained_model=$TRAINED_MODEL
```
<div align="center">
    <img src="./deploy/images/ILSVRC2012_val_00020010.jpeg" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 246, 211, 209, 208], 'scores': [0.98832, 0.00448, 0.00279, 0.00184, 0.0005], 'file_name': './deploy/images/ILSVRC2012_val_00020010.jpeg', 'label_names': ['Weimaraner', 'Great Dane', 'vizsla, Hungarian pointer', 'Chesapeake Bay retriever', 'Labrador retriever']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.98832`。

## 5. 模型推理部署

### 5.1 基于Inference的推理

可以参考[模型导出](./docs/zh_CN/inference_deployment/export_model.md)，

将该模型转为 inference 模型只需运行如下命令：

```shell
export TRAINED_MODEL=ckpt_and_log/300epoch
python tools/export_model.py \
    -c ppcls/configs/ImageNet/PeleeNet/PeleeNet.yaml\
    -o Global.save_inference_dir=./deploy/models/PeleeNet_infer \
    -o Global.pretrained_model=$TRAINED_MODEL
```

### 5.2 基于Serving的服务化部署

Serving部署教程可参考：[链接](./deploy/paddleserving/readme.md)。


## 6. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [TIPC: test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/config/PeleeNet/PeleeNet_train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/config/PeleeNet/PeleeNet_train_infer_python.txt 'lite_train_lite_infer'
```
TIPC结果：

如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到`test_tipc/output/`文件夹中的文件中。

```
Run successfully with command - python3 -m paddle.distributed.launch --gpus=0,1 train.py --lr=0.001 --data-path=./lite_data --device=cpu --output-dir=./test_tipc/output/norm_train_gpus_0,1_autocast_null --epochs=1     --batch-size=1    !  
 ...
Run successfully with command - python3 deploy/py_inference/infer.py --use-gpu=False --use-mkldnn=False --cpu-threads=6 --model-dir=./test_tipc/output/norm_train_gpus_0_autocast_null/ --batch-size=1     --benchmark=False     > ./test_tipc/output/python_infer_cpu_usemkldnn_False_threads_6_precision_null_batchsize_1.log 2>&1 !  
```

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
[Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/pdf/1804.06882.pdf)

[PeleeNet](https://github.com/Robert-JunWang/PeleeNet)