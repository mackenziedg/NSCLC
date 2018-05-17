import os
import pickle

import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import tensorflow as tf

# Define several helper functions for processing the image data
def get_path_contents(path):
    path = path[:-1] if path.endswith("/") else path
    contents = [path + "/" + f for f in os.listdir(path)]
    return contents

def get_dicoms(path):
    contents = get_path_contents(path)
    dicoms = [pydicom.read_file(f) for f in contents if f.endswith(".dcm")]
    return dicoms

def is_image_folder(path):
    return len(os.listdir(path)) > 1

def normalize(arr):
#     arr = arr.copy() # Not needed unless you also need the original array
    return (arr - arr.min())/(arr.max()-arr.min())

imwidth, imheight, imdepth = 64, 64, 64

def process_image_folder(path, vol_height=imdepth):
    # Gets the middle 60 scans for each patient and normalizes them
    dicoms = get_dicoms(path)
    dicoms = sorted(dicoms, key=lambda x: x.SliceLocation)
    ld = len(dicoms)
    bottom = ld // 2 - vol_height//2
    top = ld // 2 + vol_height//2
    dicoms = dicoms[bottom:top]
    image_vector = normalize(np.array([di.pixel_array[100:400,100:400] for di in dicoms]))
    image_vector = resize(image_vector, [imwidth,imheight,imdepth],
                          mode="constant")
    return image_vector

kernel_size = 5
pool_size = 2
strides = 2
dense_units = 128
batch_size = 50

# Define a generic 3d cnn from https://www.tensorflow.org/tutorials/layers
def cnn_model_fn(features, labels, mode):

    input_layer = tf.cast(tf.reshape(features, [-1, 1, imdepth, imwidth, imheight]), tf.float32)

    conv1 = tf.layers.conv3d(
            inputs=input_layer,
            data_format="channels_first",
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling3d(inputs=conv1, data_format="channels_first", pool_size=pool_size, strides=strides)

    conv2 = tf.layers.conv3d(
            inputs=pool1,
            data_format="channels_first",
            filters=64,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling3d(inputs=conv2, data_format="channels_first", pool_size=pool_size, strides=strides)

    pool2_flat = tf.reshape(pool2, [-1, np.prod(pool2.get_shape().as_list()[1:])])

    dense = tf.layers.dense(inputs=pool2_flat, units=dense_units, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=(mode==tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Build the training data into the correct format

ds_root = "/media/gpudata_backup/mac/NSCLC/data/img/"

#print("Getting image directories")
img_dirs = np.ravel([[[ssd for ssd in get_path_contents(sd) if is_image_folder(ssd)]
                       for sd in get_path_contents(d)]
                       for d in get_path_contents(ds_root)])

#print("Reading in/processing metatdata csv")
df = pd.read_csv("/media/gpudata_backup/mac/NSCLC/data/Lung1.clinical.balanced.histology.csv")
dx = df[~df.Histology.isnull()][["PatientID","Histology"]]
dx.loc[:,"hist_code"] = pd.Categorical(dx.Histology).codes

img_dirs = [i for i in img_dirs if i.split("/")[7] in dx.PatientID.values]

dx.loc[:, "dir_path"] = sorted(img_dirs, key=lambda x: int(x.split("/")[7][-3:]))

# Split the data
np.random.seed(1)
train, test = train_test_split(dx, test_size=0.1)

print("Reading in training images")
X_train = np.array([process_image_folder(i) for i in train.dir_path])
X_train = []
print("There are {} training images".format(train.dir_path.shape[0]))
for ix, path in enumerate(train.dir_path):
   X_train.append(process_image_folder(path))
   if ix % 20 == 0:
       print("{}/{}".format(ix, train.shape[0]))

X_train_label = train.hist_code.values
print(train.hist_code.value_counts(normalize=True))
# with open("/media/gpudata_backup/mac/NSCLC/data/X_train_images.pickle", "wb") as f:
#    pickle.dump([X_train, X_train_label], f)

print("Reading in test images.")
X_test = np.array([process_image_folder(i) for i in test.dir_path])
X_test_label = test.hist_code.values

# with open("/media/gpudata_backup/mac/NSCLC/data/X_test_images.pickle", "wb") as f:
#    pickle.dump([X_test, X_test_label], f)

#print("Initializing the classifier")
# Init. the classifier
nsclc_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="/tmp/nsclc_3d_convnet_model")

# test_samp = train.iloc[11:20,:]

# Set up logging for predictions
# validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#     X_test,
#     X_test_label,
#     every_n_steps=50)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log,
    every_n_iter=10)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X_train,
    y=X_train_label,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)

nsclc_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X_test,
    y=X_test_label,
    shuffle=False)

preds = nsclc_classifier.eval(
    input_fn=predict_input_fn,
    hooks=[logging_hook])
