import torch
import numpy as np

np.random.seed(45)
torch.manual_seed(45)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EEG Data
train_data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/eeg_raw_cvpr2017_dataset/train/*'
val_data_path   = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/eeg_raw_cvpr2017_dataset/val/*'
test_data_path  = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/eeg_raw_cvpr2017_dataset/test/*'

# Non EEG Data
data_path = '/home/prajwal_15/Prajwal/Learn/Lab/GAN/Pytorch/dataset/images_dataset/100-shot-panda/*'

image_height = 128#224
image_width  = 128#224
input_channel= 3
kernel_size  = 3
padding      = 1
batch_size   = 32
num_workers  = 8
latent_dim   = 128
diff_augment_policies = "color,translation"
lr           = 3e-4
EPOCH        = 4096