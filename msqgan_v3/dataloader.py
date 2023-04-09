import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import config
import numpy as np
import cv2

class EEG2ImageDataset(Dataset):
	def __init__(self, dataset_path, mode='train'):
		self.dataset_path = dataset_path

		self.eegs   = []
		self.images = []
		self.labels = []

		print('loading {} dataset...'.format(mode))
		for path in tqdm(natsorted(glob(dataset_path))):
			loaded_array = np.load(path, allow_pickle=True)
			self.eegs.append(loaded_array[1].T)
			img = np.float32(cv2.resize(loaded_array[0], (config.image_height, config.image_width)))
			img = (img - 127.5) / 127.5
			self.images.append(np.transpose(img, (2, 0, 1)))
			self.labels.append(loaded_array[2])

		self.eegs   = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
		self.images = torch.from_numpy(np.array(self.images)).to(torch.float32)
		self.labels = torch.from_numpy(np.array(self.labels)).to(torch.int32)


	def __len__(self):
		return self.eegs.shape[0]

	def __getitem__(self, idx):
		eeg   = self.eegs[idx]
		norm  = torch.max(eeg) / 2.0
		eeg   =  ( eeg - norm ) / norm
		image = self.images[idx]
		label = self.labels[idx]

		return eeg, image, label


class ImageDataset(Dataset):
	def __init__(self, dataset_path, mode='train'):
		self.dataset_path = dataset_path
		self.images_path  = list(natsorted(glob(dataset_path)))

	def __len__(self):
		return len(self.images_path)

	def __getitem__(self, idx):
		img = cv2.cvtColor(cv2.imread(self.images_path[idx], 1), cv2.COLOR_BGR2RGB)
		img = np.float32(cv2.resize(img, (config.image_height, config.image_width)))
		img = (img - 127.5) / 127.5
		img = np.transpose(img, (2, 0, 1))
		# img = torch.from_numpy(np.load(self.images_path[idx], allow_pickle=True), device=torch.device(config.device)).to(torch.float32)
		return img




if __name__ == '__main__':

	eeg_train_dataset = EEG2ImageDataset(dataset_path=config.train_data_path, mode='train')
	eeg_val_dataset   = EEG2ImageDataset(dataset_path=config.val_data_path, mode='val')
	eeg_test_dataset  = EEG2ImageDataset(dataset_path=config.test_data_path, mode='test')

	print(len(eeg_train_dataset), len(eeg_val_dataset), len(eeg_test_dataset))

	eeg_train_dataloader = DataLoader(eeg_train_dataset,\
									  batch_size=config.batch_size,\
									  shuffle=True,\
									  num_workers=config.num_workers,\
									  drop_last=False
									 )
	eeg_val_dataloader   = DataLoader(eeg_val_dataset,\
									  batch_size=config.batch_size,\
									  shuffle=False,\
									  num_workers=config.num_workers,\
									  drop_last=False
									 )
	eeg_test_dataloader  = DataLoader(eeg_test_dataset,\
									  batch_size=config.batch_size,\
									  shuffle=False,\
									  num_workers=config.num_workers,\
									  drop_last=False
									 )

	tq = tqdm(eeg_train_dataloader)
	for idx, (eegs, images, labels) in enumerate(tq):
		eegs, images, labels = eegs.to(config.device),\
							   images.to(config.device),\
							   labels.to(config.device)