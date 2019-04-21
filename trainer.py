import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
import argparse

from tqdm import tqdm

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras.optimizers import Adam

from models import init_transform_model, init_loss_models, init_model
from loss_functions import calc_style_loss, calc_content_loss
from batch_generators import batch_generator



def trainer(alpha, beta, image_dim, batch_size, steps_per_epoch, epochs, coco_dirpath, save_weights_path, style_image_path):
	""" Train transfomation network to reconstruct input images in the desired style """

	# Layers for loss calculations
	content_layers = ['block3_conv3']
	style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

	content_losses = [calc_content_loss]
	style_losses = [calc_style_loss, calc_style_loss, calc_style_loss, calc_style_loss]

	content_weights = [alpha]
	style_weights = [beta, beta, beta, beta]


	# Initiate models
	transform_model = init_transform_model()
	loss_model, content_loss_model, style_loss_model = init_loss_models(content_layers, style_layers)
	model = init_model(transform_model, loss_model)

	# Load style image and extract featuremaps
	style_image = cv2.imread(style_image_path).astype(np.float32) / 255
	style_image_resized = cv2.resize(style_image, (image_dim, image_dim))
	style_image_batch = np.repeat(np.expand_dims(style_image_resized, axis=0), repeats=batch_size, axis=0)

	style_featuremaps = style_loss_model.predict(style_image_batch)


	# Generator to train reconet
	batch_gen = batch_generator(coco_dirpath, content_loss_model, style_featuremaps, image_dim, batch_size)
	(a, b) = next(batch_gen) # Tensorflow bug: will crash without calling the generator once before training


	model.compile(optimizer='adam', loss=content_losses+style_losses, loss_weights=content_weights+style_weights)

	model.fit_generator(
		generator=batch_gen,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs)

	transform_model.save_weights(save_weights_path)




if __name__ == '__main__':

	# Input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--alpha')
	parser.add_argument('--beta')
	parser.add_argument('--image_dim')
	parser.add_argument('--batch_size')
	parser.add_argument('--steps_per_epoch')
	parser.add_argument('--epochs')
	parser.add_argument('--coco_dirpath')
	parser.add_argument('--save_weights_path')
	parser.add_argument('--style_image_path')
	args = parser.parse_args()
	
	# Call the trainer function with parameters
	trainer(
		alpha = int(args.alpha),
		beta = int(args.beta),
		image_dim = int(args.image_dim),
		batch_size = int(args.batch_size),
		steps_per_epoch = int(args.steps_per_epoch),
		epochs = int(args.epochs),
		coco_dirpath = args.coco_dirpath,
		save_weights_path = args.save_weights_path,
		style_image_path = args.style_image_path)