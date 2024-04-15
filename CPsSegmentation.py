"""
Author: zhangyong
email: zhangyong7630@163.com
"""
import os
import gc
import time
import datetime
import numpy as np
import torch
from monai.utils import set_determinism
from torch.optim import AdamW
from monai.data import (CacheDataset, ThreadDataLoader, set_track_meta, pad_list_data_collate, Dataset)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, TverskyLoss

from monai.transforms import (EnsureChannelFirstd, Compose, EnsureTyped,
                              LoadImaged, Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged,
                              Spacingd, SpatialPadd, RandZoomd, RandFlipd,
                              RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
                              RandAdjustContrastd, Resized)

import nvtx
from monai.utils.nvtx import Range
import contextlib


def range_func(x, y): return Range(x)(y) if profiling else y


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
profiling = False
no_profiling = contextlib.nullcontext()

# 训练相关的设置
from torch.backends import cudnn

cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
import os
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

root_dir = r'E:\dataset\paper\CPFirst\CPSegFold\PaperUpload\logs'  # 用于保存训练模型
high_hu, low_hu = 800, 0  # 窗口调整

lr = 1e-4
num_class = 2
epoch_save_flag = False
device = torch.device("cuda:0")
resume = None
train_dir = r'E:\dataset\paper\CPFirst\Fold2\train'  # 训练文件地址
val_dir = r'E:\dataset\paper\CPFirst\Fold2\val'  # 测试文件地址

val_interval = 5  # 每隔多少轮测试一次
max_epochs, save_epoch_name = 600, 'seg32'  # 最大迭代epoch 数目
origin_name, label_name = '_origin.nii.gz', '_cp.nii.gz'  # 原始图像和标签的名字， (A01_origin.nii.gz, A01_cp.nii.gz)
train_start, train_end = 0, 50
spatial_size, sw_batch_size, batch_size = (128, 128, 128), 1, 4


def dice(pre, mask, smooth=0.00001):
    # 定义dice 损失函数
    union = np.sum(pre * mask) + smooth
    score = (2 * union) / (np.sum(pre) + np.sum(mask) + smooth)
    return score


def train(save_name, model=None, resume=None, fast=True, cache=True) -> object:
    global root_dir, val_dir

    # 模型转到device
    if model is not None:
        model.to(device)
        if resume is not None:
            dict = torch.load(resume)
            model.load_state_dict(dict, strict=False)
            print("restore params successfully!")

    print(str(datetime.datetime.now()), str(f"this task is : {label_name}"))
    log_dir = os.path.join(root_dir, save_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    set_determinism(0)  # 每次的训练结果可以重复

    best_metric, best_metric_epoch, cycle_start = 0, 0, 0

    # 定义训练和测试操作
    train_trans = Compose([LoadImaged(keys=["image", "label", ], reader='ITKReader'),
                           EnsureChannelFirstd(keys=["image", "label"]),
                           SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
                           ScaleIntensityRanged(keys=["image"], a_min=low_hu, a_max=high_hu, b_min=0.0,
                                                b_max=1.0, clip=True),
                           RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=spatial_size,
                                                  pos=3, neg=1, num_samples=sw_batch_size),

                           RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10, ),
                           RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10, ),
                           RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10, ),
                           RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.2, mode=("trilinear", "nearest"),
                                     align_corners=(True, None), prob=0.15, ),
                           RandGaussianNoised(keys=["image"], prob=0.1, mean=0, std=0.1),

                           RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.25),
                           RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.1),
                           RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
                           EnsureTyped(keys=["image", "label"], track_meta=False, ),
                           ])
    val_trans = Compose([LoadImaged(keys=["image", "label", ], reader='ITKReader'),
                         EnsureChannelFirstd(keys=["image", "label"]),

                         SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
                         ScaleIntensityRanged(keys=["image"], a_min=low_hu, a_max=high_hu, b_min=0.0,
                                              b_max=1.0, clip=True),
                         EnsureTyped(keys=["image", "label"], track_meta=False),
                         ])

    files = sorted(os.listdir(train_dir))
    images, labels = [], []

    for f in files:
        image_path = os.path.join(train_dir, f, str(f) + origin_name)
        label_path = os.path.join(train_dir, f, str(f) + label_name)
        if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            labels.append(label_path)
        else:
            print(f, 'error')

    train_files = [{"image": image, "label": label} for image, label in zip(images, labels)]

    # 获取测试数据集的列表
    files = sorted(os.listdir(val_dir))
    images, labels, lungs = [], [], []

    for f in files:
        image_path = os.path.join(val_dir, f, str(f) + origin_name)
        label_path = os.path.join(val_dir, f, str(f) + label_name)
        if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            labels.append(label_path)
        else:
            print(f, 'error')

    val_files = [{"image": image, "label": label} for image, label in zip(images, labels)]

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_trans, cache_rate=1.0, num_workers=4,
                                copy_cache=False, )
        gc.collect()
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True,
                                        collate_fn=pad_list_data_collate, buffer_size=10)
    else:
        # 加载数据, 将全部数据缓存到内存中进行处理
        train_ds = Dataset(data=train_files, transform=train_trans)
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True,
                                        collate_fn=pad_list_data_collate, buffer_size=10)

    val_ds = CacheDataset(data=val_files, transform=val_trans, cache_rate=1.0, num_workers=4, copy_cache=False)
    gc.collect()
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, collate_fn=pad_list_data_collate)

    # 定义损失函数

    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, batch=True,
                               lambda_dice=1.0, lambda_ce=1.0, )

    # 优化器
    optimizer = AdamW(model.parameters(), lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 学习速率优化
    set_track_meta(False)

    count = 0
    for epoch in range(max_epochs):
        gc.collect()

        epoch_start = time.time()

        with nvtx.annotate("epoch", color="red") if profiling else no_profiling:
            torch.cuda.empty_cache()
            model.train()
            epoch_loss = 0
            train_loader_iterator = iter(train_loader)
            for step in range(1, len(train_loader) + 1):

                batch_data = next(train_loader_iterator)
                inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device),)
                # print(inputs.size())
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():

                    main_outputs = model(inputs)
                    loss = loss_function(main_outputs, labels)  # + t1_loss(main_outputs, labels)
                if fast:
                    # 采用半精度进行模型的优化
                    # profiling: backward
                    with nvtx.annotate("backward", color="blue") if profiling else no_profiling:
                        scaler.scale(loss).backward()
                    # profiling: update
                    with nvtx.annotate("update", color="yellow") if profiling else no_profiling:
                        scaler.step(optimizer)
                        scaler.update()

                epoch_loss += loss.item()
                count = count + 1
            scheduler.step()
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} step time: {(time.time() - epoch_start):.4f}")
        # 进行模型的测试
        if (epoch + 1) % val_interval == 0:
            torch.cuda.empty_cache()
            model.eval()
            out_list = []
            with torch.no_grad():
                val_loader_iterator = iter(val_loader)
                for _ in range(len(val_loader)):
                    val_data = next(val_loader_iterator)
                    val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device),)
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, spatial_size, sw_batch_size, model)
                        val_outputs = torch.argmax(val_outputs, dim=1)
                        val_outputs = val_outputs.cpu().numpy()
                        val_labels = val_labels.cpu().numpy()
                        score = dice(val_outputs, val_labels)
                        out_list.append(score)
                out_list = np.array(out_list)
                metric_mean = np.mean(out_list)

                if metric_mean > best_metric:
                    best_metric = metric_mean
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join(log_dir, str(save_name) + "_best_model_" + ".pt")
                    torch.save(model.state_dict(), save_path)

                print(
                    f"current epoch: {epoch + 1},"
                    f" mean dice: {metric_mean:.4f},"
                    f" best mean dice: {best_metric:.4f},"
                    f" at epoch: {best_metric_epoch},"
                )

                torch.cuda.empty_cache()
                print(str(datetime.datetime.now()))

    print(str(datetime.datetime.now()))

    return None


if __name__ == "__main__":
    from model.FRNet import FRNet

    model = FRNet(init_filters=32)

    train(save_name="FRNet32", fast=True, model=model, resume=resume)
