import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import imageio
import argparse

from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D

from models import init_transform_model


def video_inference(weights_path, content_video_path, combination_video_path):
	""" Generate a video of combination frames """

	# Load model graph and weights
	transform_model = init_transform_model()
	transform_model.load_weights(weights_path)

	# Initilise video reader and writer
	reader = imageio.get_reader(content_video_path)
	writer = imageio.get_writer(combination_video_path, fps=reader.get_meta_data()['fps'])

	# Iterate through input content frames, generating coresponding output combination frames.
	# for frame_index in tqdm(range(reader.get_length())):
	for frame_index in tqdm(range(500)):

		content_image = reader.get_data(frame_index)
		content_image_rescaled = content_image.astype(np.float32) / 255

		combination_image = transform_model.predict(np.expand_dims(content_image_rescaled, axis=0))[0]
		combination_image_rescaled = (combination_image * 255).astype(np.uint8)

		writer.append_data(combination_image_rescaled)

	writer.close()


if __name__ == '__main__':

	# Input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights_path')
	parser.add_argument('--content_video_path')
	parser.add_argument('--combination_video_path')
	args = parser.parse_args()

	video_inference(
		weights_path=args.weights_path,
		content_video_path=args.content_video_path,
		combination_video_path=args.combination_video_path)
