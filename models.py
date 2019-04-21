import tensorflow as tf

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, UpSampling2D, add
from tensorflow.keras.optimizers import Adam

from instancenormalization import InstanceNormalization


def residual_block(filters, kernel_size=(3,3)):

	residual_input = Input((None, None, filters))
	residual_conv1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same')(residual_input)
	residual_conv2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same')(residual_conv1)
	residual_output = add([residual_input, residual_conv2])

	residual_model = Model(inputs=residual_input, outputs=residual_output)

	return residual_model


def init_transform_model():
	""" Model architecture to deconstruct input image, then reconstruct an output image """

	# Encoder
	encoder_model = Sequential()
	encoder_model.add(Conv2D(filters=48, kernel_size=(9,9), strides=(1,1), padding='same', input_shape=(None,None,3)))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), padding='same'))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(2,2), padding='same'))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	# Residual blocks x4
	encoder_model.add(residual_block(filters=192))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(residual_block(filters=192))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(residual_block(filters=192))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(residual_block(filters=192))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())


	# Decoder
	decoder_model = Sequential()

	decoder_model.add(UpSampling2D(size=(2,2), input_shape=(None,None,192)))
	decoder_model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same'))
	decoder_model.add(InstanceNormalization(axis=-1))
	decoder_model.add(ReLU())

	decoder_model.add(UpSampling2D(size=(2,2)))
	decoder_model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding='same'))
	decoder_model.add(InstanceNormalization(axis=-1))
	decoder_model.add(ReLU())

	decoder_model.add(Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), padding='same', activation='sigmoid'))


	# Full Model
	model_input = Input((None,None,3))
	model_encoder = encoder_model(model_input)
	model_decoder = decoder_model(model_encoder)
	model = Model(inputs=model_input, outputs=model_decoder)

	return model


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


def init_model(transform_model, loss_model):
	""" Append tranformation and loss models"""

	model_input = Input((None,None,3))
	model_transform = transform_model(model_input)
	model_loss = loss_model(model_transform)

	model = Model(inputs=model_input, outputs=model_loss)

	return model


if __name__ == '__main__':

	model = init_transform_model()

	print(model.summary())