import os
import time

import numpy as np
from medpy import metric
import SimpleITK as sitk
from inference_utlis import get_model, save_nii_from_numpy, inference_monai, get_transform_resave, val_loader



def CPSegMulti(test_dir, model, log_path, device, out_name):
    files = os.listdir(test_dir)

    # 加载模型
    model = get_model(model, log_path)
    dice_list, Pr_list, Re_list, HD_list = [], [], [], []

    inference_list = []
    deep_spatial_size = (128, 128, 128)
    for f in files:

        nii_path = os.path.join(test_dir, f, str(f) + "_origin.nii.gz")
        mask_path = os.path.join(test_dir, f, str(f) + "_cp.nii.gz")
        if os.path.exists(nii_path) and os.path.exists(mask_path):
            a1 = time.time()
            img = sitk.ReadImage(nii_path)
            deep_trans = get_transform_resave(low_hu=0, high_hu=800, spatial_size=deep_spatial_size)
            deep_loader = val_loader(image_path=nii_path, label_path=mask_path, trans=deep_trans)

            pre, label, origin_image = inference_monai(model, deep_loader, deep_spatial_size, sw_batch_size=4,
                                                       device=device, )

            # 保存结果

            save_path = os.path.join(test_dir, f, str(f) + out_name)
            save_nii_from_numpy(img, save_path, pre)  # 将结果保存为nii.gz 文件
            a2 = time.time()
            inference_list.append(a2 - a1)
            # 保存标签和原始图像, 这是因为MR图像涉及空间变换，monai 处理后可能存在空间位置不对。
            label_out_path = os.path.join(test_dir, f, str(f) + "_prelabel.nii.gz")

            if not os.path.exists(label_out_path):
                save_nii_from_numpy(img, label_out_path, label)

            origin_out_path = os.path.join(test_dir, f, str(f) + "_preorigin.nii.gz")
            if not os.path.exists(origin_out_path):
                save_nii_from_numpy(img, origin_out_path, origin_image, label_flag=False)

            # 计算相关的指标
            dice_score = metric.binary.dc(pre, label)  # 计算dice 系数
            precision = metric.binary.precision(pre, label)
            recall = metric.binary.recall(pre, label)
            hd = metric.binary.hd95(pre, label)
            print(f, ',', dice_score, ',', precision, ',', recall, ',', hd)

            dice_list.append(dice_score)
            Pr_list.append(precision)
            Re_list.append(recall)
            HD_list.append(hd)

    inference_list = np.array(inference_list)
    dice_list = np.array(dice_list)
    print("Dice", '%.4f' % np.mean(dice_list), "±" '%.4f' % np.std(dice_list))
    Pr_list = np.array(Pr_list)
    print("Pr", '%.4f' % np.mean(Pr_list), "±" '%.4f' % np.std(Pr_list))
    Re_list = np.array(Re_list)
    print("Re", '%.4f' % np.mean(Re_list), "±" '%.4f' % np.std(Re_list))
    HD_list = np.array(HD_list)
    print("HD", '%.4f' % np.mean(HD_list), "±" '%.4f' % np.std(HD_list))

    print('inference speed:', deep_spatial_size, np.mean(inference_list), np.std(inference_list))

    return None


if __name__ == "__main__":
    import torch
    import sys
    current_dir = sys.path[0]
    device = torch.device("cuda:0")
    from model.FRNet import FRNet
    model = FRNet(init_filters=32)
    test_dir = r'E:\dataset\paper\CPFirst\Fold2\val'
    log_path = os.path.join(current_dir, "logs", 'FRNet32_Fold2.pt')
    out_name = 'FRNet.nii.gz'

    CPSegMulti(test_dir, model, log_path, device, out_name)
