import random

import torch
from torchvision.transforms import *
from PIL import Image
import math
from config.config import cfg
from torchvision.transforms import functional as F
from torchvision.transforms import _functional_pil as F_pil
class RandomErasing(object):
	""" Randomly selects a rectangle region in an image and erases its pixels.
		'Random Erasing Data Augmentation' by Zhong et al.
		See https://arxiv.org/pdf/1708.04896.pdf
	Args:
		p: The prob that the Random Erasing operation will be performed.
		sl: Minimum proportion of erased area against input image.
		sh: Maximum proportion of erased area against input image.
		r1: Minimum aspect ratio of erased area.
		mean: Erasing value.
	"""
	def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
		self.p = p
		self.mean = mean
		self.sl = sl
		self.sh = sh
		self.r1 = r1

	def __call__(self, img):
		if random.uniform(0, 1) >= self.p:
			return img

		for attempt in range(100):
			area = img.size()[1] * img.size()[2]

			target_area = random.uniform(self.sl, self.sh) * area
			aspect_ratio = random.uniform(self.r1, 1 / self.r1)

			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))

			if w < img.size()[2] and h < img.size()[1]:
				x1 = random.randint(0, img.size()[1] - h)
				y1 = random.randint(0, img.size()[2] - w)
				if img.size()[0] == 3:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
					img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
					img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
				else:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
				return img

		return img


class ChannelRandomErasing(object):
	""" Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

	def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):

		self.probability = probability
		self.mean = mean
		self.sl = sl
		self.sh = sh
		self.r1 = r1

	def __call__(self, img):

		if random.uniform(0, 1) > self.probability:
			return img

		for attempt in range(100):
			area = img.size()[1] * img.size()[2]

			target_area = random.uniform(self.sl, self.sh) * area
			aspect_ratio = random.uniform(self.r1, 1 / self.r1)

			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))

			if w < img.size()[2] and h < img.size()[1]:
				x1 = random.randint(0, img.size()[1] - h)
				y1 = random.randint(0, img.size()[2] - w)
				if img.size()[0] == 3:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
					img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
					img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
				else:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
				return img

		return img


class ChannelAdapGray(object):
	""" Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

	def __init__(self, probability=0.5):
		self.probability = probability

	def __call__(self, img):

		# if random.uniform(0, 1) > self.probability:
		# return img

		idx = random.randint(0, 3)

		if idx == 0:
			# random select R Channel
			img[1, :, :] = img[0, :, :]
			img[2, :, :] = img[0, :, :]
		elif idx == 1:
			# random select B Channel
			img[0, :, :] = img[1, :, :]
			img[2, :, :] = img[1, :, :]
		elif idx == 2:
			# random select G Channel
			img[0, :, :] = img[2, :, :]
			img[1, :, :] = img[2, :, :]
		else:
			if random.uniform(0, 1) > self.probability:
				# return img
				img = img
			else:
				tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
				img[0, :, :] = tmp_img
				img[1, :, :] = tmp_img
				img[2, :, :] = tmp_img
		return img


class ChannelExchange(object):
	""" Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

	def __init__(self, gray=2):
		self.gray = gray

	def __call__(self, img):

		idx = random.randint(0, self.gray)

		if idx == 0:
			# random select R Channel
			img[1, :, :] = img[0, :, :]
			img[2, :, :] = img[0, :, :]
		elif idx == 1:
			# random select B Channel
			img[0, :, :] = img[1, :, :]
			img[2, :, :] = img[1, :, :]
		elif idx == 2:
			# random select G Channel
			img[0, :, :] = img[2, :, :]
			img[1, :, :] = img[2, :, :]
		else:
			tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
			img[0, :, :] = tmp_img
			img[1, :, :] = tmp_img
			img[2, :, :] = tmp_img
		return img

class RectScale(object):
	def __init__(self, height, width, interpolation=Image.BILINEAR):
		self.height = height
		self.width = width
		self.interpolation = interpolation

	def __call__(self, img):
		w, h = img.size
		if h == self.height and w == self.width:
			return img
		return img.resize((self.width, self.height), self.interpolation)


class GrayMix(torch.nn.Module):
	def __init__(self, num_output_channels=3,patch_size=16,prob=0.7):
		super(GrayMix,self).__init__()
		self.num_output_channels = num_output_channels
		self.patch_size=patch_size
		self.prob=prob

	def forward(self, img):
		"""
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Grayscaled image.
        """
		c,h,w=img.shape
		p = []
		numx=h // self.patch_size
		numy=w // self.patch_size
		for i in range(numx):
			for j in range(numy):
				p.append([i * self.patch_size, j * self.patch_size])
		# print(p)
		# plen=len(p)
		mask=torch.rand(numx*numy)<self.prob
		for idx,(i, j) in enumerate(p):
			if mask[idx]:
				img[:, i:i + self.patch_size, j:j + self.patch_size]=F.rgb_to_grayscale(img[:, i:i + self.patch_size, j:j + self.patch_size],
								   num_output_channels=self.num_output_channels)
		return img
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transform_mix_aug = [transforms.ColorJitter(brightness=0.3,contrast=0.3),
					 transforms.GaussianBlur(21, sigma=(0.1, 3))
					 ]

transform_rgb2gray = transforms.Compose([
		transforms.ToPILImage(),
		RectScale(cfg.H, cfg.W),
		transforms.RandomHorizontalFlip(),
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		normalize,
		RandomErasing(p=0.5)
    ])

transform_thermal = transforms.Compose([
		transforms.ToPILImage(),
		RectScale(cfg.H, cfg.W),
		transforms.RandomHorizontalFlip(),
		transforms.RandomChoice(transform_mix_aug),
		transforms.ToTensor(),
		normalize,
		RandomErasing(p=0.5)
    ])


transform_rgb = transforms.Compose([
		transforms.ToPILImage(),
		RectScale(cfg.H, cfg.W),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
		RandomErasing(p=0.5)
	])


transform_test = transforms.Compose([
	transforms.ToPILImage(),
	RectScale(cfg.H, cfg.W),
	transforms.ToTensor(),
	normalize
])


transform_sysu = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((cfg.H, cfg.W)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(p = 0.5, sl = 0.2, sh = 0.8, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_regdb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((cfg.H, cfg.W)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(p = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])

# transform_regdb1 = transforms.Compose([
# 	transforms.ToPILImage(),
# 	transforms.Pad(10),
# 	transforms.RandomCrop((cfg.H, cfg.W)),
# 	transforms.RandomHorizontalFlip(),
# 	# transforms.RandomGrayscale(p = 0.1),
# 	transforms.ToTensor(),
# 	normalize,
# 	ChannelRandomErasing(probability=0.5)])
#
# transform_regdb2 = transforms.Compose([
# 	transforms.ToPILImage(),
# 	transforms.Pad(10),
# 	transforms.RandomCrop((cfg.H, cfg.W)),
# 	transforms.RandomHorizontalFlip(),
# 	transforms.ToTensor(),
# 	normalize,
# 	ChannelRandomErasing(probability=0.5),
# 	ChannelExchange(gray=2)])



transform_regdbtogray = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((cfg.H, cfg.W)),
    transforms.RandomHorizontalFlip(),
	transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    normalize,
    RandomErasing(p = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
])
transform_regdbtogray_grayMix = transforms.Compose([
	transforms.ToPILImage(),
	RectScale(cfg.H, cfg.W),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	GrayMix(3,16,0.0),#transforms.Grayscale(num_output_channels=3),
	normalize,
	RandomErasing(p=0.5)
])
