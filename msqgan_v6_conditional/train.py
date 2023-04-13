import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import config
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import numpy as np
from dataloader import EEG2ImageDataset, ImageDataset
from save_figure import vis
from diff_augment import DiffAugment
from losses import disc_hinge, gen_hinge
from network_multi_disc import ImageGAN
import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.device)

def train_step_dis(model, optimizer, images, noise_1, con=None):
	
	num_disc       = 0.0
	dis_loss       = 0.0
	gen_image      = model.gen(noise_1, con)
	optimizer['dis'].zero_grad()
	gen_image_aug  = DiffAugment(gen_image, policy=config.diff_augment_policies)
	images_aug     = DiffAugment(images, policy=config.diff_augment_policies)
	dis_score_fake_lst = model.dis(gen_image_aug, con)
	dis_score_real_lst = model.dis(images_aug, con)
	for dis_score_real, dis_score_fake in zip(dis_score_real_lst, dis_score_fake_lst):
		# print(dis_score_real.shape, dis_score_fake.shape)
		dis_loss       += disc_hinge(dis_score_real, dis_score_fake)
		# num_disc       += 1
	# dis_loss/=num_disc
	dis_loss.backward()
	optimizer['dis'].step()

	return dis_loss.item()


def train_step_gen(model, optimizer, images, noise_1, noise_2, con=None):

	gen_loss = 0.0
	
	optimizer['gen'].zero_grad()
	gen_image_1 = model.gen(noise_1, con)
	gen_image_2 = model.gen(noise_2, con)

	gen_image_1_aug = DiffAugment(gen_image_1, policy=config.diff_augment_policies)
	gen_image_2_aug = DiffAugment(gen_image_2, policy=config.diff_augment_policies)

	dis_score_fake_1_lst = model.dis(gen_image_1_aug, con)
	dis_score_fake_2_lst = model.dis(gen_image_2_aug, con)

	for dis_score_fake_1, dis_score_fake_2 in zip(dis_score_fake_1_lst, dis_score_fake_2_lst):
		gen_loss        += ( gen_hinge(dis_score_fake_1) + gen_hinge(dis_score_fake_2) )
	# for dis_score_fake_1 in dis_score_fake_1_lst:
	# 	gen_loss += gen_hinge(dis_score_fake_1)

	# gen_loss   = gen_hinge(dis_score_fake_1) + gen_hinge(dis_score_fake_2)
	mode_loss  = torch.divide(torch.mean(torch.abs(torch.subtract(gen_image_2, gen_image_1))),\
							 torch.mean(torch.abs(torch.subtract(noise_2, noise_1))))
	# total_loss = 2.0 * gen_loss #+ 0.02 * torch.mean(loss_fn_vgg(images, gen_image_1)) #+ 0.2 * mode_loss
	total_loss = 0.9 * gen_loss + 0.1 * mode_loss
	# total_loss = 1.0 * gen_loss #+ torch.mean(loss_fn_vgg(images, gen_image_1)) + torch.mean(torch.square(torch.subtract(images, gen_image_1)))
	total_loss.backward()
	optimizer['gen'].step()

	return total_loss.item()


def train(model, optimizer, epoch, dataloader):
	
	tq       = tqdm(dataloader)
	dis_loss = []
	gen_loss = []
	for idx, (eegs, images, labels, cons) in enumerate(tq):
		# eegs      = eegs.to(config.device)
		images, labels, cons = images.to(config.device), labels.to(config.device), cons.to(config.device)
		noise_1  = ((-1.0-1.0) * torch.rand(size=(images.shape[0], config.latent_dim)) + 1.0 ).to(config.device)
		noise_2  = ((-1.0-1.0) * torch.rand(size=(images.shape[0], config.latent_dim)) + 1.0 ).to(config.device)
		# noise_1  = (torch.rand(size=(images.shape[0], config.latent_dim))).to(config.device)
		# noise_2  = (torch.rand(size=(images.shape[0], config.latent_dim))).to(config.device)
		# noise_1  = (torch.normal(mean=0.0, std=1.0, size=(images.shape[0], config.latent_dim))).to(config.device)
		# noise_2  = (torch.normal(mean=0.0, std=1.0, size=(images.shape[0], config.latent_dim))).to(config.device)
		dis_loss.append( train_step_dis(model, optimizer, images, noise_1, cons) )
		gen_loss.append( train_step_gen(model, optimizer, images, noise_1, noise_2, cons) )
		tq.set_description('e:{}, g:{}/d:{}'.format(epoch, sum(gen_loss)/len(gen_loss), sum(dis_loss)/len(dis_loss)))


if __name__ == '__main__':

	eeg_train_dataset = EEG2ImageDataset(dataset_path=config.train_data_path, mode='train')
	# eeg_val_dataset   = EEG2ImageDataset(dataset_path=config.val_data_path, mode='val')
	# eeg_test_dataset  = EEG2ImageDataset(dataset_path=config.test_data_path, mode='test')
	# image_dataset     = ImageDataset(dataset_path=config.data_path, mode='images')
	# print(len(eeg_train_dataset), len(eeg_val_dataset), len(eeg_test_dataset))

	eeg_train_dataloader = DataLoader(eeg_train_dataset,\
									  batch_size=config.batch_size,\
									  shuffle=True,\
									  num_workers=config.num_workers,\
									  drop_last=False,\
									  pin_memory=True
									 )
	# eeg_val_dataloader   = DataLoader(eeg_val_dataset,\
	# 								  batch_size=config.batch_size,\
	# 								  shuffle=False,\
	# 								  num_workers=config.num_workers,\
	# 								  drop_last=False,\
	# 								  pin_memory=True
	# 								 )
	# eeg_test_dataloader  = DataLoader(eeg_test_dataset,\
	# 								  batch_size=config.batch_size,\
	# 								  shuffle=False,\
	# 								  num_workers=config.num_workers,\
	# 								  drop_last=False,\
	# 								  pin_memory=True
	# 								 )


	# image_dataloader     = DataLoader(image_dataset,\
	# 								  batch_size=config.batch_size,\
	# 								  shuffle=True,\
	# 								  num_workers=config.num_workers,\
	# 								  drop_last=False,\
	# 								  pin_memory=True
	# 								 )

	# tq = tqdm(eeg_train_dataloader)
	# for idx, (eegs, images, labels) in enumerate(tq):
	# 	eegs, images, labels = eegs.to(config.device),\
	# 						   images.to(config.device),\
	# 						   labels.to(config.device)
	model     = ImageGAN(image_size=config.image_height, c=config.n_classes).to(config.device)
	model.gen = torch.nn.DataParallel(model.gen).to(config.device)
	model.dis = torch.nn.DataParallel(model.dis).to(config.device)
	optimizer = {
					# 'gen': torch.optim.AdamW(\
					# 						list(model.gen.parameters()),\
					# 						lr=config.lr,\
					# 						betas=(config.beta_1, config.beta_2)),
					# 'dis': torch.optim.AdamW(\
					# 						list(model.dis.parameters()),\
					# 						lr=config.lr,\
					# 						betas=(config.beta_1, config.beta_2))
					'gen': torch.optim.Adam(\
											list(model.gen.parameters()),\
											lr=config.lr,\
											betas=(config.beta_1, config.beta_2)),
					'dis': torch.optim.Adam(\
											list(model.dis.parameters()),\
											lr=config.lr,\
											betas=(config.beta_1, config.beta_2))
				}

	dir_info  = natsorted(glob('EXPERIMENT_*'))
	
	if len(dir_info)==0:
		experiment_num = 1
	else:
		experiment_num = int(dir_info[-1].split('_')[-1]) + 1

	ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/imagegan_*.pth'.format(experiment_num)))

	START_EPOCH = 0

	if len(ckpt_lst)>=1:
		ckpt_path  = ckpt_lst[-1]
		checkpoint = torch.load(ckpt_path, map_location=config.device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer['gen'].load_state_dict(checkpoint['optimizer_state_dict_gen'])
		optimizer['dis'].load_state_dict(checkpoint['optimizer_state_dict_dis'])
		init_noise = checkpoint['init_noise']
		init_con   = checkpoint['init_con']
		init_label = checkpoint['init_label']
		# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		START_EPOCH = checkpoint['epoch']
		print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
		START_EPOCH += 1
	else:
		init_noise = ((-1.0-1.0)*torch.rand(size=(config.batch_size, config.latent_dim))+1.0).to(config.device)
		_, _, init_label, init_con = next(iter(eeg_train_dataloader))
		init_label, init_con = init_label.to(config.device), init_con.to(config.device)
		# init_noise = (torch.normal(mean=0.0, std=1.0,size=(config.batch_size, config.latent_dim))).to(config.device)
		os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
		os.makedirs('EXPERIMENT_{}/train/'.format(experiment_num))
		os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

	for epoch in range(START_EPOCH, config.EPOCH):
		train(model, optimizer, epoch, eeg_train_dataloader)
		vis(model, epoch, init_noise, init_con, init_label, experiment_num)

		torch.save({
		'epoch': epoch,
		'init_noise': init_noise,
		'init_con': init_con,
		'init_label': init_label,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict_gen': optimizer['gen'].state_dict(),
		'optimizer_state_dict_dis': optimizer['dis'].state_dict(),
		# 'scheduler_state_dict': scheduler.state_dict(),
		}, 'EXPERIMENT_{}/checkpoints/imagegan_{}.pth'.format(experiment_num, 'all'))
