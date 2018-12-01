import tensorflow as tf
import numpy as np

'''
Potential things to change:
- number of filters in convolutional layers
- number of channels outside of input layer (see math in notebook)
- pool size in pooling layers (reduces dimensions of next layer by that factor)
- dropout rate
- activation function
- node units in dense layer
'''


def cnn_model_functions(features, labels, mode, lr=0.001):

	# transform images into 64x64 numpy arrays
	features = pre_process(features)

	# create input layer
	input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

	# TODO: check filter number; currently maintaining 1:1 then 2:1
	con_layer1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 64,
		kernel_size = [5, 5],
		padding="same",
		activation = tf.nn.relu)

	pool_layer1 = tf.layers.max_pooling2d(
		inputs=con_layer1,
		pool_size=[4,4], 
		strides=4)

	con_layer2 = tf.layers.conv2d(
		inputs = pool_layer1,
		filters = 128,
		kernel_size = [5,5],
		padding = "same",
		activation = tf.nn.relu)

	pool_layer2 = tf.layers.max_pooling2d(
		inputs=con_layer2,
		pool_size=[2,2], 
		strides=2)

	# Make pooling layer one dimensional
	# TODO: there are 3 channels in input due to color image; 
	# is that relationship maintained in future layer channels?
	pool_flat = tf.reshape(pool_layer2,
		[-1, 8 * 8 * 128 * 3])

	# create fully connected layer
	dense = tf.layers.dense(
		inputs=pool_flat,
		units=1024,
		activation=tf.nn.relu)

	# perform dropout while training to avoid overfitting
	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training = mode == tf.estimator.ModeKeys.TRAIN)

	# output layer
	logits = tf.layers.dense(
		inputs=dropout,
		units=14)

	# create a map of predictions for PREDICT and EVAL modes
	predictions = {
		"classes": tf.argmax(inputs=logits, axis=1)
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# return the prediction if we're in the prediction mode
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# otherwise, calculate loss and train some schtuff
	loss = tf.losses.sparse_softmax_cross_entropy(
		labels=labels,
		logits=logits)


	# training
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# evaluating
	else:
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=predictions["classes"])
		}

		return tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			eval_metric_ops=eval_metric_ops)

def pre_process(features):

	for feature in features:


	return final_features