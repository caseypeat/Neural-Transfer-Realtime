import numpy as np
import tensorflow as tf


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