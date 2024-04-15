# Feature Reserve Network for Choroid Plexus Segmentation in 7T MRI
## Methods
In this paper, we proposed a novel Feature Reserve Network (FRNet) that integrates detail reserve (DR) module, semantic reserve (SR) module and  feature aggregation (FA) module for Choroid Plexus (CPs) segmentation in 7T MRI. Unlike methods solely focused on enhancing semantic feature extraction for optimal network performance, FRNet aims to effectively retain and propagate critical features throughout the network.
![本地路径](figure/Architecture.png") <!-- 此路径表示图片和MD文件，处于同一目录 -->
## Usage
### Installation
Install the necessary python packages as defined in environment.txt. We recommend using conda. You can create the environment using：

`pip install requirements.txt `

---

### Data Preparation
please Prepare dataset in the following manner:
- dataset
 - - train
 
The CPs segmentation dataset (CPs-170) is publicly available, please contact us if needed：wei chen, email: landcv@tmmu.edu.cn


---
### Training
For default training configuration, we use patch-based training pipeline and use AdamW optimizer. The patch-size is 128×128×128 with batch size of 4. 


```
python CPsSegmentation.py --model FRNet
```


---

### Inference
If you want to test the model which has been trained on the CPs-170 dataset, run the inference script as following:

`python CPSegMulti.py --test_dir, --model, --log_path, --device, --out_name`

We provide ckpt download via Google Drive or Baidu Netdisk. Please download the checkpoint in the inference stage from the url below:

**Google Drive**

url：https://drive.google.com/drive/folders/1kxO2AVsc_5atRA8iO4U9GyKfd01HA0Oj

**Baidu Netdisk**

url：https://pan.baidu.com/s/14Z-h_lFd-PB6ePz2HXEb0A?pwd=dsdu extraction code (提取码)：dsdu 

---




