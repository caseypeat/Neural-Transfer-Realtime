from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras.optimizers import Adam


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

	decoder_model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same'))
	decoder_model.add(BatchNormalization(axis=-1))
	decoder_model.add(ReLU())

	decoder_model.add(UpSampling2D(size=(2,2)))

	decoder_model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding='same'))
	decoder_model.add(BatchNormalization(axis=-1))
	decoder_model.add(ReLU())

	decoder_model.add(Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), padding='same', activation='tanh'))

	# ReCoNet
	reconet_input = Input((None,None,3))
	reconet_encoder = encoder_model(reconet_input)
	reconet_decoder = decoder_model(reconet_encoder)
	reconet_model = Model(inputs=reconet_input, outputs=reconet_decoder)

	return reconet_model