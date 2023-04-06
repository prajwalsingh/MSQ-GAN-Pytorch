import config
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# https://inf.news/en/tech/ef67a9c98d34e20c1c718239908d9399.html
class PixelNorm(nn.Module):
	def __init__(self, epsilon=1e-8):
		super(PixelNorm, self).__init__()
		self.epsilon = epsilon

	def forward(self, x):
		return torch.divide(x, torch.sqrt( torch.mean( x ** 2, axis=-1, keepdims=True ) + self.epsilon ))


class Generator(nn.Module):

	def __init__(self, c=None):
		super(Generator, self).__init__()

		if c is None:
			c = 0

		in_channels    = [128+c, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
		out_channels   = [ 1024,  512, 256, 128,  64, 32, 16,  8, 4]
		scale          = [    4,    2,   2,   2,   2,  2,  2,  2, 2]
		self.depth     = int(math.log2(config.image_height)) - 1

		self.layers = nn.ModuleList()

		for idx in range(0, self.depth):

			self.layers.append( nn.Sequential( 
								nn.Upsample(scale_factor=scale[idx], mode='nearest'),
								nn.utils.spectral_norm(
									nn.Conv2d(in_channels=in_channels[idx],\
											  out_channels=out_channels[idx],\
											  kernel_size=config.kernel_size,\
											  stride=1,\
											  padding=config.padding,
											  bias=False
										    )),
								nn.BatchNorm2d(num_features=out_channels[idx]),
								nn.LeakyReLU()
								))

		self.layers.append(nn.Sequential(nn.utils.spectral_norm(
									nn.Conv2d(in_channels=in_channels[idx+1],\
											    out_channels=3,\
											    kernel_size=config.kernel_size,\
											    stride=1,\
											    padding=config.padding,
											    bias=False
											)),
									nn.Tanh()
									))

	def forward(self, x, c=None):
		if c is not None:
			pass

		x = torch.unsqueeze(torch.unsqueeze(x, dim=-1), dim=-1)

		for layer in self.layers:
			# print(x.shape)
			x = layer( x )

		return x
		

class Discriminator(nn.Module):

	def __init__(self, c=None):
		super(Discriminator, self).__init__()

		if c is None:
			c = 0

		in_channels    = [config.input_channel+c,   16,  32,  64,  128, 256,  512, 1024]
		out_channels   = [                    16,   32,  64, 128,  256, 512, 1024,    1]
		stride         = [                     2,    2,   2,   2,    2,   2,    2,    1]
		total_depth    =  8
		start_depth    =  (total_depth+1) - (int(math.log2(config.image_height)) - 1)

		self.layers = nn.ModuleList()

		in_channels[start_depth] = config.input_channel+c

		for idx in range(start_depth, total_depth):

			self.layers.append( nn.Sequential( 
									nn.Conv2d(in_channels=in_channels[idx],\
											  out_channels=out_channels[idx],\
											  kernel_size=config.kernel_size,\
											  stride=stride[idx],\
											  padding=config.padding,
											  bias=False
										    ),
								nn.BatchNorm2d(num_features=out_channels[idx]),
								nn.LeakyReLU()
								))

		self.out = nn.Linear(
								in_features=1*8*8,
								out_features=1
							)

	def forward(self, x, c=None):
		if c is not None:
			pass

		for layer in self.layers:
			# print(x.shape)
			x = layer( x )
		# print(x.shape)
		x = self.out( torch.reshape(x, (x.shape[0], -1)) )
		return x


class ImageGAN(nn.Module):

	def __init__(self, c=None):
		super(ImageGAN, self).__init__()
		self.gen = Generator(c=c)
		self.gen.apply(weights_init)
		self.dis = Discriminator(c=c)
		self.dis.apply(weights_init)


if __name__ == '__main__':

	noise     = torch.randn(size=(config.batch_size, 128)).to(config.device)
	x_fake    = torch.randn(size=(config.batch_size, 3, config.image_height, config.image_width)).to(config.device)
	gan_model = ImageGAN().to(config.device)
	gen_image = gan_model.gen(noise)
	print(gen_image.shape, torch.min(gen_image), torch.max(gen_image))
	dis_label = gan_model.dis(x_fake)
	print(dis_label.shape, torch.min(dis_label), torch.max(dis_label))