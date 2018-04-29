from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

					
def card_classify_fn(features, labels, mode):
	"""CNN model to detect a card"""
	# input layer
	input_layer = tf.reshape(features["x"], [-1, 57, 57, 3])
	
	# conv1
	# input shape [batch_size, 57, 57, 3]
	# output shape [batch_size, 57, 57, 32]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu)
	
	# pool1
	# output shape [batch_size, 19, 19, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], stride=3)
	
	# pool2
	# output shape [batch_size, 19, 19, 64]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)
	
	# pool2
	# output shape [batch_size, 9, 9, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], stride=2)
	
	pool2_flat = tf.reshape(pool2, [-1, 9 * 9 * 64])
	
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout, unit=4)
	
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
		
	# calculate loss
	loss = tf.loess.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizier(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loess,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
		
	eval_metrics_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metrics_ops=eval_metrics_ops)
					
def main(argv):
	print(data_loader.train_input_fn)
	args = parser.parse_args(argv[1:])

	# Create Estimator
	classifier = tf.estimator.Estimator(
		model_fn = card_classify_fn, model_dir="D:\TF\card")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	# Train the Model.
	classifier.train(
		input_fn=lambda: data_loader.train_input_fn,
		steps=args.train_steps,
		hooks=[logging_hook])

	# TODO: create real eval and Evaluate the model.
	eval_result = classifier.evaluate(
		input_fn=data_loader.train_input_fn)

	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    #expected = ['Setosa', 'Versicolor', 'Virginica']
    #predict_x = {
    #    'SepalLength': [5.1, 5.9, 6.9],
    #    'SepalWidth': [3.3, 3.0, 3.1],
    #    'PetalLength': [1.7, 4.2, 5.4],
    #    'PetalWidth': [0.5, 1.5, 2.1],
    #}

    #predictions = classifier.predict(
    #   input_fn=lambda:iris_data.eval_input_fn(predict_x,
    #                                            labels=None,
    #                                           batch_size=args.batch_size))

    #template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    #for pred_dict, expec in zip(predictions, expected):
    #   class_id = pred_dict['class_ids'][0]
    #    probability = pred_dict['probabilities'][class_id]

    #   print(template.format(iris_data.SPECIES[class_id],
    #                          100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)