import torch
import math
import numpy as np

np.random.seed(45)
torch.manual_seed(45)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EEG Data
train_data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/eeg_raw_cvpr2017_dataset/train/*'
val_data_path   = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/eeg_raw_cvpr2017_dataset/val/*'
test_data_path  = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/eeg_raw_cvpr2017_dataset/test/*'

# Non EEG Data
# data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/100-shot-panda/*'
data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/eeg_imagenet_40_cvpr_2017_data/images_imagenet40_cvpr_2017/train/*/*'
# data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/numpy_ImageNet10/*/*'

image_height = 128#224
image_width  = 128#224
input_channel= 3
kernel_size  = 3
padding      = 1
batch_size   = 256#128
num_workers  = 8
latent_dim   = 128
diff_augment_policies = "color,translation,cutout"
lr           = 3e-4
beta_1       = 0.2
beta_2       = 0.5
EPOCH        = 2051#1024
ckpt_freq    = 10
num_col      = int(2 * math.log2(batch_size))