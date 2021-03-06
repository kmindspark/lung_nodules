from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import os
import imageio as ii
import numpy as np
import pandas as pd
import dicom as dcm
import scipy as sp
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

# Constants
ROOTDIRTRAIN = "Train_Images/"
ROOTDIRTEST = "Validation_Images/"
MINROWS = 64
MINCOLS = 64

# Main program
def main(unused_argv):
  print("Starting program...")
  trainingData = pd.read_csv("train_files.csv")
  testData = pd.read_csv("validation_files.csv")

  train_images = np.empty((len(trainingData.index), MINROWS, MINCOLS))
  train_labels = np.empty((len(trainingData.index)))
  for index, row in trainingData.iterrows():
    curImg = dcm.read_file(ROOTDIRTRAIN + str(row[0]), cv2.IMREAD_GRAYSCALE)
    #print(ROOTDIRTRAIN + str(row[0]))
    train_images[index] = curImg.pixel_array #.flatten()
    train_images[index] /= train_images[index].max()
    print(train_images[index])
    train_labels[index] = row[1]

  test_images = np.empty((len(testData.index), MINROWS, MINCOLS))
  test_labels = np.empty((len(testData.index)))
  for index, row in testData.iterrows():
    curImg = dcm.read_file(ROOTDIRTEST + str(row[1]), cv2.IMREAD_GRAYSCALE)
    #print(ROOTDIRTEST + str(row[1]))
    test_images[index] = curImg.pixel_array #.flatten()
    test_images[index] /= test_images[index].max()
    print(test_images[index])
    test_labels[index] = row[2]

  train_images = train_images.astype('float32')
  test_images = test_images.astype('float32')

  train_labels = np.asarray(train_labels, dtype=np.int32)
  test_labels = np.asarray(test_labels, dtype=np.int32)

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(log_step_count_steps=1)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_images},
    y=train_labels,
    batch_size=20, #15, #64
    num_epochs=None,
    shuffle=True)

  eval_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_images},
    y=test_labels,
    num_epochs=1,
    shuffle=False)

  breast_cancer_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, config=run_config)#,
  #model_dir="tmp/convnet_model")

  tensors_to_log = {"probabilities": "softmax_tensor"}

  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)

  #breast_cancer_classifier = tf.contrib.estimator.add_metrics(breast_cancer_classifier, accuracy_metric)

  print("Training started...")

  experiment = tf.contrib.learn.Experiment(estimator=breast_cancer_classifier, train_input_fn=train_input_fn,eval_input_fn=eval_input_fn_2,train_steps=25000,eval_steps=None,eval_delay_secs=0,train_steps_per_iteration=1000)

  """
  breast_cancer_classifier.train(
    input_fn=train_input_fn,
    steps=25000,#500,#500,#1500,#4000,
    hooks=[logging_hook])
  """

  experiment.continuous_train_and_eval()

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_images},
    y=train_labels,
    num_epochs=1,
    shuffle=False)

  """
  eval_results = breast_cancer_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  eval_results_2 = breast_cancer_classifier.evaluate(input_fn=eval_input_fn_2)
  print(eval_results_2)
  """
  eval_results = experiment.evaluate()
  print(eval_results)

  print("Program completed.")


def cnn_model_fn(features, labels, mode):
  print("Working...")

  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, MINROWS, MINCOLS, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,#8
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.variance_scaling_initializer)


  conv12 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.variance_scaling_initializer)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,#16
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.variance_scaling_initializer)

  conv22 = tf.layers.conv2d(
      inputs=conv2,
      filters=64,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.variance_scaling_initializer)
  pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2)

  """
  # Convolutional Layer #3 and Pooling Layer #3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  """

  # Dense Layer
  pool3_flat = tf.reshape(pool2, [-1, 256 * 64])
  dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu) #1024
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.02, training=mode == tf.estimator.ModeKeys.TRAIN) #0.6,0.4,0.1 (latest)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  #class_weights = tf.constant([0.8, 0.2])
  class_weights = tf.constant([0.755813953, 0.244186047])
  #class_weights = tf.constant([0.507, 0.493])
  weights = tf.gather(class_weights, labels)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
  #l1_loss = tf.add_n([tf.nn.l1_loss(v) for v in tf.trainable_variables()])
  l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
  #mylambda1 = 0.2
  mylambda2 = 0.005 # 0.1 0.5 #0.4 (latest) 0.2 0.1
  loss = loss + mylambda2 * l2_loss
    
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005) #0.0001
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    auc = tf.metrics.auc(labels=labels, predictions=predictions["classes"])
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy[1], "auc" : auc[1]}, every_n_iter=1) #, "auc" : auc[1], ", "accuracy" : accuracy[1]"

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

  sess = tf.Session()

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "overall_accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]
      ),
      "overall_accuracy_mean": tf.metrics.mean_per_class_accuracy(
          labels=labels, predictions=predictions["classes"], num_classes=2),
      "overall_precision": tf.metrics.precision(
          labels=labels, predictions=predictions["classes"]
      ),
      "overall_recall": tf.metrics.recall(
          labels=labels, predictions=predictions["classes"]
      ),
      "auc": tf.metrics.auc(
          labels=labels, predictions=predictions["classes"]
      ),
      "label_0": tf.metrics.percentage_below(
          values=labels, threshold=1
      ),
      "pred_0": tf.metrics.percentage_below(
          values=predictions["classes"], threshold=1
      )}
      #"accuracy_0": class_accuracy(labels, predictions["classes"], 0),
      #"accuracy_1": class_accuracy(labels, predictions["classes"], 1),
      #"accuracy_2": class_accuracy(labels, predictions["classes"], 2)}

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def class_accuracy(labels,preds,num):
  totCount = 0
  hitCount = 0
  for i, val in enumerate(labels):
    if val == num:
      totCount = totCount + 1
      if val == preds[i]:
        hitCount = hitCount + 1
  return hitCount/totCount


if __name__ == "__main__":
  tf.app.run()
