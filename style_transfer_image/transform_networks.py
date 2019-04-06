import tensorflow as tf

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, add, ZeroPadding2D, Layer
from tensorflow.keras.optimizers import Adam

from instancenormalization import InstanceNormalization


class ReflectPadding2D(ZeroPadding2D):

	def __init__(self, padding=(1, 1)):
		super(ReflectPadding2D, self).__init__()
		self.padding = ((0,0), padding, padding, (0,0))

	def call(self, x):
		return tf.pad(x, self.padding, "REFLECT")


def residual_block(filters, kernel_size=(3,3)):

	residual_input = Input((None, None, filters))
	residual_conv1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same')(residual_input)
	residual_conv2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same')(residual_conv1)
	residual_output = add([residual_input, residual_conv2])

	residual_model = Model(inputs=residual_input, outputs=residual_output)

	return residual_model


def init_transform_network():
	"""  """

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
	decoder_model.add(ReflectPadding2D(padding=(13,13)))


	# Full Model
	model_input = Input((None,None,3))
	model_encoder = encoder_model(model_input)
	model_decoder = decoder_model(model_encoder)
	model = Model(inputs=model_input, outputs=model_decoder)

	return model


if __name__ == '__main__':

	model = init_transform_network()

	print(model.summary())