import tensorflow as tf
import numpy as np
from scipy import misc
import glob

TRAIN_DIR = "..\card_*.png"
TEST_DIR = "..\card_*.png"

# need to create as dataset to easy shuffle and augument data
# maybe just a linear classifer
# https://stackoverflow.com/questions/44933096/read-images-into-tensorflow
def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	train_file_list = glob.glob(TRAIN_DIR)
	a = [tf.image.resize_image_with_crop_or_pad(tf.image.decode_png(x, 3), 57, 57) for x in train_file_list]
	features = np.array(a)
	labels = np.array(['1hdsc'.find(x[8]) for x in train_file_list])
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(500).repeat().batch(batch_size)

	# Return the dataset.
	return dataset

	#a = [tf.image.resize_image_with_crop_or_pad(tf.image.decode_png(x, 3), 57, 57) for x in train_file_list]
	#return tf.estimator.inputs.numpy_input_fn(
	#	{"x" : np.array(a)},
	#	np.array(['1hdsc'.find(x[8]) for x in train_file_list]),
	#	batch_size = 128,
	#	num_epochs = 3,
	#	shuffle = True)

# TODO: create real eval func
def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def main():
	train_file_list = glob.glob(TRAIN_DIR)
	a = [tf.image.resize_image_with_crop_or_pad(tf.image.decode_png(x, 3), 57, 57) for x in train_file_list]
	features = np.array(a)
	labels = np.array(['1hdsc'.find(x[8]) for x in train_file_list])
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	#t = []
	#for img in train_file_list:
	#	a = tf.image.resize_image_with_crop_or_pad(tf.image.decode_png(img, 3), 57, 57)
	#	t.append(a)
	#train_data = np.array(t)
	#labels = np.array(['1hdsc'.find(x[8]) for x in train_file_list])
	#dataset = tf.data.Dataset.from_tensor_slices(train_data, labels)
	
	
if __name__ == '__main__':
	main()