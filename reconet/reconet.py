from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, add
from tensorflow.keras.optimizers import Adam

from instancenormalization import InstanceNormalization


def residual_block(filters, kernel_size):

	residual_input = Input((None, None, filters))
	residual_conv1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same')(residual_input)
	residual_conv2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding='same')(residual_conv1)
	residual_output = add([residual_input, residual_conv2])

	residual_model = Model(inputs=residual_input, outputs=residual_output)

	return residual_model


def init_reconet_model():
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
	encoder_model.add(residual_block(filters=192, kernel_size=(3,3)))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(residual_block(filters=192, kernel_size=(3,3)))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(residual_block(filters=192, kernel_size=(3,3)))
	encoder_model.add(InstanceNormalization(axis=-1))
	encoder_model.add(ReLU())

	encoder_model.add(residual_block(filters=192, kernel_size=(3,3)))
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

	# ReCoNet
	reconet_input = Input((None,None,3))
	reconet_encoder = encoder_model(reconet_input)
	reconet_decoder = decoder_model(reconet_encoder)
	reconet_model = Model(inputs=reconet_input, outputs=reconet_decoder)

	return reconet_model


