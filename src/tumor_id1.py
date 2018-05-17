import json
import os

import numpy as np
import pydicom
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.set_printoptions(threshold=np.nan)
# Global defines (it's not hacky at all)
img_width, img_height = 120, 120
kernel_size = 5
pool_size = 2
stride = 2
batch_size = 100

def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features, [-1, 1, img_width, img_height])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        data_format="channels_first",
        filters=128,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        data_format="channels_first",
        pool_size=pool_size,
        strides=stride)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        data_format="channels_first",
        filters=64,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        data_format="channels_first",
        pool_size=pool_size,
        strides=stride)
    pool2_flat = tf.reshape(pool2, [-1, tf.cast(img_width*img_height//(pool_size**4)*64, tf.int32)])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.9, training=(mode==tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid)
    preds = tf.cast(tf.round(logits), tf.int32)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.log_loss(labels=labels, predictions=logits)

    predictions = {
        "class": tf.round(logits),
        "probability": tf.identity(logits, name="logit"),
        "loss": tf.identity(loss, name="loss")
    }

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"class": tf.round(logits),
                       "probability": tf.identity(logits, name="probability")}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions)

    true_negatives = tf.metrics.true_negatives(labels=labels,
                                               predictions=predictions["class"])
    false_negatives = tf.metrics.false_negatives(labels=labels,
                                               predictions=predictions["class"])
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["class"]),
        "true_neg": tf.metrics.true_negatives(labels=labels,
                                               predictions=predictions["class"]),
        "false_neg": tf.metrics.false_negatives(labels=labels,
                                               predictions=predictions["class"]),
        "true_pos": tf.metrics.true_positives(labels=labels,
                                               predictions=predictions["class"]),
        "false_pos": tf.metrics.false_positives(labels=labels,
                                               predictions=predictions["class"])}
    # loss = tf.Print(loss, [tf.confusion_matrix(tf.cast(tf.round(logits), tf.int32),
                           # tf.reshape(labels, [-1]), num_classes=2)])
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
    return (arr - arr.min())/(arr.max()-arr.min())


def process_image_folder(path, tumor_locs):
    # Gets the middle 60 scans for each patient and normalizes them
    dicoms = get_dicoms(path)
    dicoms = sorted(dicoms, key=lambda x: x.SliceLocation)
    ld = len(dicoms)
    image_vector = np.array([di.pixel_array[100:400,100:400] for di in dicoms])
    for sl, img in zip([i.SliceLocation for i in dicoms], image_vector):
        i = normalize(resize(img, [img_width,img_height], mode="constant"))
        if sl in tumor_locs:
            id_ = 1
        else:
            id_ = 0

        for j in range(4):
            yield np.rot90(i, j), id_
            yield np.rot90(np.fliplr(i), j), id_
            yield np.rot90(np.flipud(i), j), id_

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create the Estimator
    nsclc_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/nsclc_convnet_model",
    warm_start_from="/tmp/nsclc_convnet_model")

    with open("/media/gpudata_backup/mac/NSCLC/data/has_tumors.json", "r") as f:
        has_tumors = json.load(f)

    ds_root = "/media/gpudata_backup/mac/NSCLC/data/img/"

    counts = [0,0]

    def generate_data(paths, eval=False):
        for path in paths:
            id_ = path.split("/")[7]
            if id_ in has_tumors.keys():
                p = process_image_folder(path, has_tumors[id_])

                for pair in p:
                    if pair[1] == 0:
                        if not eval:
                            if np.random.random() < 0.12:
                                counts[0] += 1
                                yield pair
                        else:
                            counts[0] += 1
                            yield pair
                    else:
                        counts[1] += 1
                        yield pair

    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=1000)

    img_dirs = np.ravel([[[ssd for ssd in get_path_contents(sd) if is_image_folder(ssd)]
                           for sd in get_path_contents(d)]
                           for d in get_path_contents(ds_root)])
    input_size = sum([len(os.listdir(i)) for i in img_dirs])

    np.random.seed(1)
    train, test = train_test_split(img_dirs, test_size=0.1)
    test_size = sum([len(os.listdir(i)) for i in test])

    def generate_train():
        return generate_data(train)

    def generate_eval():
        return generate_data(test, eval=True)

    def train_input_fn():
        data = tf.data.Dataset.from_generator(generate_train,
                                              (tf.float32, tf.int32))
        data = data.repeat(5).shuffle(10000)
        return data

    def eval_input_fn():
        data = tf.data.Dataset.from_generator(generate_eval,
                                              (tf.float32, tf.int32))
        return data.batch(200)

    # nsclc_classifier.train(input_fn=train_input_fn,
    #                        hooks=[logging_hook])

    # nsclc_classifier.predict(eval_input_fn)
    nsclc_classifier.evaluate(eval_input_fn)
