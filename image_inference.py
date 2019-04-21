import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D

from models import init_transform_model


def image_inference(weights_path, content_image_path):
	""" Generate a combination image and display it"""

	# Load model graph and weights
	transform_model = init_transform_model()
	transform_model.load_weights(weights_path)

	# Load content image
	content_image = cv2.imread(content_image_path).astype(np.float32) / 255

	# Generate combination image from content image using the style transform model
	combination_image = transform_model.predict(np.expand_dims(content_image, axis=0))

	# Visualise content, and combination images side by side 
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB))
	ax[1].imshow(cv2.cvtColor(combination_image[0], cv2.COLOR_BGR2RGB))
	plt.show()


if __name__ == '__main__':

	# Input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights_path')
	parser.add_argument('--content_image_path')
	args = parser.parse_args()

	image_inference(
		weights_path=args.weights_path,
		content_image_path=args.content_image_path)