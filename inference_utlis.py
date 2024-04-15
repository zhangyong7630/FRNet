"""
Author: zhangyong
email: zhangyong7630@163.com
"""

import SimpleITK as sitk
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, SpatialPadd,
                              EnsureTyped)
from monai.data import ThreadDataLoader, Dataset


def inference_monai(model, val_loader, spatial_size, sw_batch_size, device):
    with torch.cuda.amp.autocast():
        model.eval()

        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                val_inputs = val_data["image"].to(device)

                origin_image, val_label = val_data["image1"], val_data["label"]
                val_outputs = sliding_window_inference(val_inputs, spatial_size, sw_batch_size, model)

                # 将tensor 转换为矩阵， 留了一个接口，便于后期的批量操作
                val_outputs = torch.argmax(val_outputs, dim=1)
                val_outputs = val_outputs.cpu().numpy()
                val_outputs = np.squeeze(val_outputs)
                save_data = np.array(val_outputs, dtype=np.float32)
                val_label = np.squeeze(val_label)
                origin_image = np.squeeze(origin_image)

                return save_data, val_label, origin_image


def get_transform_resave(low_hu, high_hu, spatial_size):
    # 对数据进行预处理操作
    trans = Compose([LoadImaged(keys=["image", "image1", "label"], reader='ITKReader'),
                     EnsureChannelFirstd(keys=["image", "image1", "label"]),
                     SpatialPadd(keys=["image", "image1", "label"], spatial_size=spatial_size),

                     ScaleIntensityRanged(keys=["image"], a_min=low_hu, a_max=high_hu, b_min=0.0,
                                          b_max=1.0, clip=True),
                     EnsureTyped(keys=["image", "label"]),
                     ])
    return trans


def val_loader(image_path, label_path, trans):
    val_dicts = [{"image": image_path, "image1": image_path, "label": label_path}]
    val_dataset = Dataset(data=val_dicts, transform=trans)
    val_dataloader = ThreadDataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)

    return val_dataloader


def save_nii_from_numpy(img, save_path, save_data, label_flag=True):
    save_data = np.transpose(save_data, axes=(0, 2, 1))
    if label_flag:
        save_data = np.array(save_data, dtype='uint8')
    savedImg = sitk.GetImageFromArray(save_data)
    # 将预测结果写入到文件种
    savedImg.SetOrigin(img.GetOrigin())
    savedImg.SetDirection(img.GetDirection())
    savedImg.SetSpacing(img.GetSpacing())
    sitk.WriteImage(savedImg, save_path)
    return save_path


def get_model(model, log_path, device=torch.device("cuda:0")):
    # 加载模型
    model.to(device)
    if log_path is not None:
        model.load_state_dict(torch.load(log_path, map_location='cuda:0'), strict=False)  # 加载模型
    return model
