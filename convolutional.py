import tensorflow as tf
import numpy as np
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

'''
Potential things to change:
- number of filters in convolutional layers
- number of channels outside of input layer (see math in notebook)
- pool size in pooling layers (reduces dimensions of next layer by that factor)
- dropout rate
- activation function
- node units in dense layer
'''

IMG_RES = 64

LABEL_DICT = {'animal': 0, 'blank': 1, 'building': 2, 'child': 3, 'figure': 4, 'group': 5, 'mythical': 6, 'object': 7, 'plants': 8, 'portrait': 9, 'portrait_female': 10, 'portrait_male': 11, 'symbol': 12, 'text': 13}
INV_DICT = {v: k for k, v in LABEL_DICT.items()}

def cnn_model_functions(features, labels, mode):

	print("Starting NN")

	# create input layer
	input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

	print("Done with input layer")

	# TODO: check filter number; currently maintaining 1:1 then 2:1
	con_layer1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 32 * 3,
		kernel_size = [5, 5],
		padding="same",
		activation = tf.nn.relu)

	pool_layer1 = tf.layers.max_pooling2d(
		inputs=con_layer1,
		pool_size=[2,2], 
		strides=2)

	con_layer2 = tf.layers.conv2d(
		inputs = pool_layer1,
		filters = 64 * 3,
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
		[-1, 16 * 16 * 64 * 3])

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
		"classes": tf.argmax(input=logits, axis=1),
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
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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

# def get_data(train_folder, eval_folder):
# 	return (train_data, eval_data)

def train(train_data, train_labels, classifier, iterations=50):
	
	# log information
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=iterations)

	# train our model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=14,
		num_epochs=None,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps=4000,
		hooks=[logging_hook])

	return classifier

def test(eval_data, eval_labels, classifier):
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    	x={"x": eval_data},
    	y=eval_labels,
    	num_epochs=1,
    	shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	
	return eval_results

def main():
	train_data = []
	train_labels = []
	eval_data = []
	eval_labels = []

	# preprocess images to 64 x 64 numpy arrays
	bulk = get_data("training_set_64", "test_set_64")

	# grab data
	for key, value in bulk[0].items():
		print(len(value))
		for image in value:
			train_data.append(image)
			train_labels.append(LABEL_DICT[key])

	for key, value in bulk[1].items():
		for image in value:
			eval_data.append(image)
			eval_labels.append(LABEL_DICT[key])
		
	# print("TRAINDATA")
	# print(train_data)
	# print(len(train_data))
	# print("Trainlabels")
	# print(train_labels)
	# cast to numpy arrays
	train_data = np.asarray(train_data)
	train_labels = np.asarray(train_labels)
	eval_data = np.asarray(eval_data)
	eval_labels = np.asarray(eval_labels)

	print("---------------CREATING CLASSIFIER----------------")
	# create estimator
	coin_classifier = tf.estimator.Estimator(model_fn = cnn_model_functions, model_dir = "checkpoints/")

	print("---------------TRAINING CLASSIFIER----------------")
	# train the classifier
	coin_classifier = train(train_data, train_labels, coin_classifier)

	print("---------------EVALUATING CLASSIFIER----------------")
	# evaluate effectiveness
	results = test(eval_data, eval_labels, coin_classifier)
	print(results)

def get_data(train_folder, eval_folder):
	train_dict = {}
	for training_set in os.listdir(train_folder):
		if training_set != ".DS_Store":
			train_dict[training_set] = get_folder_elements(train_folder + "/" + training_set)
		print("Done with a sub folder")
	print("Done with Train_Dict")
	test_dict = {}
	for test_set in os.listdir(eval_folder):
		if test_set != ".DS_Store":
			test_dict[test_set] = get_folder_elements(eval_folder + "/" + test_set)
		print("done with a sub folder")


	return (train_dict, test_dict)

def get_folder_elements(folder): 
	onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

	file_names = []
	for _file in onlyfiles:
		if _file != ".DS_Store" and _file.endswith(".jpg"):
			file_names.append(_file)

	images = []

	for _file in file_names:
		print(_file)
		size = IMG_RES, IMG_RES
		im = Image.open(folder + "/" + _file)
		im_resized = im.resize(size, Image.ANTIALIAS)
		im_resized.save(folder + "/" + _file, "PNG")
		img = load_img(folder + "/" + _file)  # this is a PIL image
		# Convert to Numpy Array
		x = img_to_array(img)  
		converted_array = convert_image_to_64x64(x, len(x), len(x[0]))
		images.append(converted_array)
	for each in images:
		print(each)
	return images


def convert_image_to_64x64(array, height, width):
	num_per_bucket = np.zeros((IMG_RES, IMG_RES, 3))
	new_array = np.zeros((IMG_RES, IMG_RES, 3))
	height_ratio = height/IMG_RES
	width_ratio = width/IMG_RES


	for row in range(0, height):
		for col in range(0, width):
			for rgb in range(0, 3):
				new_array[int(row//height_ratio)][int(col//width_ratio)][rgb] += array[row][col][rgb]
				num_per_bucket[int(row//height_ratio)][int(col//width_ratio)][rgb] += 1

	for row in range(0, IMG_RES):
		for col in range(0, IMG_RES):
			for rgb in range(0, 3):
				new_array[row][col][rgb] = int(new_array[row][col][rgb]/num_per_bucket[row][col][rgb])

	return new_array

	
main()
