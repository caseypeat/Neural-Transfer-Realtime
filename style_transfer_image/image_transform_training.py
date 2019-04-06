import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time

from tqdm import tqdm

from tensorflow.train import AdamOptimizer

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras.optimizers import Adam

from transform_networks import init_transform_network

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


### LOSS FUNCTIONS
def gram_matrix(input_tensor):
	""" Calculates the gram matrix of a tensor """

	flat_featuremaps = tf.reshape(input_tensor, [tf.shape(input_tensor)[0], -1, tf.shape(input_tensor)[3]])

	gram = tf.matmul(flat_featuremaps, flat_featuremaps, transpose_a=True)

	return gram

def calc_style_loss(style, combination):
	""" Calculates the style loss between two output featuremap tensors """

	style_gram = gram_matrix(style)
	combination_gram = gram_matrix(combination)

	mse = tf.reduce_sum(tf.square(style_gram - combination_gram), axis=[-2, -1])
	normilize_constant = tf.square(tf.reduce_prod(tf.cast(tf.shape(combination)[1:], dtype=tf.float32)))

	loss = tf.divide(mse, normilize_constant)

	return loss

def calc_content_loss(content, combination):
	""" Calculates the content loss between two output featuremap tensors """

	mse = tf.reduce_sum(tf.square(content - combination), axis=[-3, -2, -1])
	normilize_constant = tf.reduce_prod(tf.cast(tf.shape(combination), dtype=tf.float32))

	loss = tf.divide(mse, normilize_constant)

	return loss


### INITIATE MODELS
def init_loss_models(content_layers, style_layers):
	""" Return base model (VGG) with outputs for layers used in loss calculations """

	base_model = VGG19(include_top=False, weights='imagenet')

	outputs = [base_model.get_layer(output_layer).output for output_layer in content_layers+style_layers]
	content_outputs = [base_model.get_layer(output_layer).output for output_layer in content_layers]
	style_outputs = [base_model.get_layer(output_layer).output for output_layer in style_layers]

	loss_model = Model(inputs=base_model.inputs, outputs=outputs)
	content_loss_model = Model(inputs=base_model.inputs, outputs=content_outputs)
	style_loss_model = Model(inputs=base_model.inputs, outputs=style_outputs)

	loss_model.trainable = False
	content_loss_model.trainable = False
	style_loss_model.trainable = False

	return loss_model, content_loss_model, style_loss_model

def init_model(transform_network, loss_model):

	model_input = Input((None,None,3))
	model_transform = transform_network(model_input)
	model_loss = loss_model(model_transform)

	model = Model(inputs=model_input, outputs=model_loss)

	return model


### BATCH GENERATORS
def batch_generator(content_dirpath, content_model, style_featuremaps, image_dim, batch_size):

	filenames = os.listdir(content_dirpath)

	while True:

		input_batch = np.zeros((batch_size, image_dim, image_dim, 3))

		batch_filenames = random.sample(filenames, batch_size)

		for i, filename in enumerate(batch_filenames):

			content_image = cv2.imread(os.path.join(content_dirpath, filename)).astype(np.float32) / 255
			content_image_resized = cv2.resize(content_image, (image_dim, image_dim))

			input_batch[i] = content_image_resized

		content_featuremaps = content_model.predict(input_batch)

		target_batch = [content_featuremaps] + style_featuremaps

		yield (input_batch, target_batch) 





if __name__ == '__main__':

	tf.enable_eager_execution()

	alpha = 1
	beta = 1
	image_dim = 256

	batch_size = 16
	steps_per_epoch = 100
	epochs = 10

	coco_dirpath = '../../Datasets/coco_unlabeled_2017'
	save_weights_path = './models/a_muse_1_256_3.h5'
	style_image_path = '../images/style_images/a_muse.jpg'


	optimizer = AdamOptimizer(learning_rate=0.001)


	# Layers for loss calculations
	content_layers = ['block3_conv3']
	style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

	content_losses = [calc_content_loss]
	style_losses = [calc_style_loss, calc_style_loss, calc_style_loss, calc_style_loss]

	content_weights = [alpha]
	style_weights = [beta, beta, beta, beta]


	# Initiate models
	transform_network = init_transform_network()
	loss_model, content_loss_model, style_loss_model = init_loss_models(content_layers, style_layers)
	model = init_model(transform_network, loss_model)

	# Load style image and extract featuremaps
	style_image = cv2.imread(style_image_path).astype(np.float32) / 255
	style_image_resized = cv2.resize(style_image, (image_dim, image_dim))
	style_image_batch = np.repeat(np.expand_dims(style_image_resized, axis=0), repeats=batch_size, axis=0)

	style_featuremaps = style_loss_model.predict(style_image_batch)


	# Generator to train reconet
	batch_gen = batch_generator(coco_dirpath, content_loss_model, style_featuremaps, image_dim, batch_size)
	(a, b) = next(batch_gen)


	model.compile(optimizer, loss=content_losses+style_losses, loss_weights=content_weights+style_weights)

	model.fit_generator(
		generator=batch_gen,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs)

	transform_network.save_weights(save_weights_path)