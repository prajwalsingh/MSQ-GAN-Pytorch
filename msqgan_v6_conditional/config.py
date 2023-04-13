import torch
import math
import numpy as np

np.random.seed(45)
torch.manual_seed(45)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EEG Data
train_data_path = '/media/cvig/A/dataset/eeg_imagenet40_cvpr_2017_raw/train/*'
val_data_path   = '/media/cvig/A/dataset/eeg_imagenet40_cvpr_2017_raw/val/*'
test_data_path  = '/media/cvig/A/dataset/eeg_imagenet40_cvpr_2017_raw/test/*'

# Non EEG Data
# data_path   = '/media/cvig/A/dataset/dogs/*'
# data_path   = '/media/cvig/A/dataset/cars/*'
# data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/cars/*'
# data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/cars/*'
# data_path = '/media/cvig/A/dataset/100-shot-panda/*'
# data_path   = '/media/cvig/A/dataset/images_dataset/eeg_imagenet_40_cvpr_2017_data/images_imagenet40_cvpr_2017/train/*/*'
# data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/numpy_ImageNet10/*/*'

image_height = 64#128#224
image_width  = 64#128#224
input_channel= 3
kernel_size  = 3
padding      = 1
batch_size   = 256
num_workers  = 8
latent_dim   = 128
n_classes    = 40
# diff_augment_policies = "color,translation,cutout"
diff_augment_policies = "color,translation"
lr           = 3e-4
beta_1       = 0.2
beta_2       = 0.5
EPOCH        = 4101#2051#1024
ckpt_freq    = 10
num_col      = 16#int(2 * math.log2(batch_size))
c_dim        = 128 # Conditinal Dimension Size
dis_level    = 2