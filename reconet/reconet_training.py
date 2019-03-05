import numpy as np
import tensorflow as tf
import cv2
import os

from tqdm import tqdm

from tensorflow.train import AdamOptimizer

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras.optimizers import Adam



### LOSS FUNCTIONS
def gram_matrix(input_tensor):
	""" Calculates the gram matrix of a tensor """

	vector = tf.reshape(input_tensor, [-1, tf.shape(input_tensor)[3]])

	gram = tf.matmul(vector, vector, transpose_a=True)

	return gram

def calc_style_loss(style, combination):
	""" Calculates the style loss between two output featuremap tensors """

	style_gram = gram_matrix(style)
	combination_gram = gram_matrix(combination)

	mse = tf.reduce_sum(tf.square(style_gram - combination_gram))
	normilize_constant = tf.square(tf.reduce_prod(tf.cast(tf.shape(combination), dtype=tf.float32)))

	loss = tf.divide(mse, normilize_constant)
	loss_reshaped = tf.expand_dims(loss, axis=0)

	return loss_reshaped

def calc_content_loss(content, combination):
	""" Calculates the content loss between two output featuremap tensors """

	mse = tf.reduce_sum(tf.square(content - combination))
	normilize_constant = tf.reduce_prod(tf.cast(tf.shape(combination), dtype=tf.float32))

	loss = tf.divide(mse, normilize_constant)
	loss_reshaped = tf.expand_dims(loss, axis=0)

	return loss_reshaped

def calc_total_loss(content_featuremaps, style_featuremaps, combination_featuremaps, alpha, beta):
	""" Calculates total loss over all style and content output featuremap tensors """

	total_loss = 0

	# Iterate over featuremaps for matching content
	for content, combination in zip(content_featuremaps, combination_featuremaps[:len(content_featuremaps)]):

		loss = calc_content_loss(content, combination) * alpha
		total_loss += loss

	# Iterate over featuremaps for matching style
	for style, combination in zip(style_featuremaps, combination_featuremaps[len(content_featuremaps):]):

		loss = calc_style_loss(style, combination) * beta
		total_loss += loss

	return total_loss


### INITIATE MODELS
def init_reconet_model():
	"""  """

	# Encoder
	encoder_model = Sequential()
	encoder_model.add(Conv2D(filters=48, kernel_size=(9,9), strides=(1,1), padding='same', input_shape=(None,None,3)))
	encoder_model.add(BatchNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), padding='same'))
	encoder_model.add(BatchNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(2,2), padding='same'))
	encoder_model.add(BatchNormalization(axis=-1))
	encoder_model.add(ReLU())

	# TODO: add residual blocks

	# Decoder
	decoder_model = Sequential()
	decoder_model.add(UpSampling2D(size=(2,2), input_shape=(None,None,192)))

	encoder_model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same'))
	encoder_model.add(BatchNormalization(axis=-1))
	encoder_model.add(ReLU())

	decoder_model.add(UpSampling2D(size=(2,2)))

	encoder_model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding='same'))
	encoder_model.add(BatchNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), padding='same', activation='sigmoid'))

	# ReCoNet
	reconet_input = Input((None,None,3))
	reconet_encoder = encoder_model(reconet_input)
	reconet_decoder = decoder_model(reconet_encoder)
	reconet_model = Model(inputs=reconet_input, outputs=reconet_decoder)

	return reconet_model

def init_loss_model(content_layers, style_layers):
	""" Return base model (VGG) with outputs for layers used in loss calculations """

	base_model = VGG19(include_top=False, weights='imagenet')

	outputs = [base_model.get_layer(output_layer).output for output_layer in content_layers+style_layers]

	loss_model = Model(inputs=base_model.inputs, outputs=outputs)

	loss_model.trainable = False

	return loss_model

def init_model(reconet_model, loss_model):

	model_input = Input((None,None,3))
	model_reconet = reconet_model(model_input)
	model_loss = loss_model(model_reconet)

	model = Model(inputs=model_input, outputs=model_loss)

	return model



if __name__ == '__main__':

	#tf.enable_eager_execution()

	optimizer = AdamOptimizer(learning_rate=0.001)

	# Layers for loss calculations
	content_layers = ['block4_conv2']
	style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

	content_losses = [calc_content_loss]
	style_losses = [calc_style_loss, calc_style_loss, calc_style_loss, calc_style_loss, calc_style_loss]

	content_weights = [1]
	style_weights = [100, 100, 100, 100, 100]

	# Initiate models
	reconet_model = init_reconet_model()
	loss_model = init_loss_model(content_layers, style_layers)
	model = init_model(reconet_model, loss_model)

	epoch_length = 10

	image_dim = 512

	# Load images
	style_image = cv2.imread('../style_images/starry_night.jpg').astype(np.float32) / 255
	style_image_resized = cv2.resize(style_image, (image_dim, image_dim))
	style_image_batch = np.zeros((epoch_length,image_dim,image_dim,3), dtype=np.float32)

	for i in range(epoch_length):
		style_image_batch[i] = style_image_resized

	style_featuremaps = loss_model.predict(style_image_batch)[len(content_layers):]

	
	coco_dirpath = '../../Datasets/coco_unlabeled_2017'
	filename_list = os.listdir(coco_dirpath)

	model.compile(optimizer, loss=content_losses+style_losses, loss_weights=content_weights+style_weights)

	for i in range(0, 10000, epoch_length):

		print('\nIteration: ', i)

		content_image_batch = np.zeros((epoch_length,image_dim,image_dim,3), dtype=np.float32)

		for i, filename in tqdm(enumerate(filename_list[i:i+epoch_length]), total=epoch_length):
			content_image = cv2.imread(os.path.join(coco_dirpath, filename)).astype(np.float32) / 255
			content_image_resized = cv2.resize(content_image, (image_dim, image_dim))

			content_image_batch[i] = content_image_resized

		content_featuremaps = loss_model.predict(content_image_batch)[:len(content_layers)]

		model.fit(x=content_image_batch, y=content_featuremaps+style_featuremaps, batch_size=1)

	reconet_model.save_weights('./models/starry_night_100.h5')