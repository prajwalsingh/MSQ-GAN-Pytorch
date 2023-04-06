import torch
import cv2
import numpy as np
import os

def save_figure(X, save_path, categ=None):
	
	X = X.detach().cpu().numpy()
	X = np.transpose(X, (0, 2, 3, 1))
	# if not os.path.isdir(save_path):
	# 	os.makedirs(os.path.join(save_path, 'train'))
	# 	os.makedirs(os.path.join(save_path, 'val'))
	# 	os.makedirs(os.path.join(save_path, 'test'))

	N      = X.shape[0]
	img_h  = X.shape[1]
	img_w  = X.shape[2]
	img_c  = X.shape[3]
	C      = 8
	R      = N // C
	h, w   = 0, 0
	canvas = np.ones((R*img_h, C*img_w, img_c), dtype=np.uint8)

	for img in X:
		img = np.uint8(np.clip(255*(img * 0.5 + 0.5), 0.0, 255.0))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		canvas[h:h+img_h, w:w+img_w, :] = img
		w += img_w
		if w>=(C*img_w):
			w  = 0
			h += img_h

	cv2.imwrite(save_path, canvas)
	return canvas

def vis(model, epoch, latent, experiment_num):
	# if not os.path.isdir(exp_dir+'/results'):
	# 	os.makedirs(exp_dir+'/results/generated/')

	# H_hat     = model.gcn(graph, H, training=False)
	# latent    = tf.concat([H_hat, latent], axis=-1)
	# latent    = tf.concat([H, latent], axis=-1)
	with torch.no_grad():
		X_fake = model.gen(latent) # GEN
	# _, X_recon = model.disc(X, training=False)

	X_fake  = save_figure(X_fake, 'EXPERIMENT_{}/train/{}.jpg'.format(experiment_num, epoch))