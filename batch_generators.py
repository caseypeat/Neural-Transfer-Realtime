import numpy as np
import tensorflow as tf
import cv2
import os
import random


def batch_generator(content_dirpath, content_model, style_featuremaps, image_dim, batch_size):
	""" Generate random batchs of input images, and corresponding target feature maps """

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