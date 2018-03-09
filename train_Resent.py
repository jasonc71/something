from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from delf import delf_v1
import time
from nets import resnet_v1

cmd_args = None
slim = tf.contrib.slim

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def list_images(directory, convert=False):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    # labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    labels = map(int, labels)

    if convert and max(labels) + 1 != len(set(labels)):
        print("***************************************************")
        print("some labels are missing, converting it")
        print("***************************************************")
        unique_labels = list(set(labels))

        label_to_int = {}
        for i, label in enumerate(unique_labels):
            label_to_int[label] = i

        labels = [label_to_int[l] for l in labels]
    return filenames, labels


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    return image, label


def read_varaibles_from_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def check_accuracy(sess, correct_prediction, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    count = 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction)
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
            if count % 100 == 0:
                acc = float(num_correct) / num_samples
                # print("finished reading " + str(num_samples) + " data")
                # print("current accuracy: " + str(acc))
            count += 1
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def build_model(images, num_classes, is_training=True, reuse=None):
    model = delf_v1.DelfV1()
    net, end_points = model.GetResnet50Subnetwork(images, global_pool=True, is_training=is_training, reuse=reuse)

    with slim.arg_scope(
            resnet_v1.resnet_arg_scope(
                weight_decay=0.0001, batch_norm_scale=True)):
        with slim.arg_scope([slim.batch_norm], is_training=True):
            feature_map = end_points['resnet_v1_50/block3']
            feature_map = slim.conv2d(
                feature_map,
                512,
                1,
                rate=1,
                activation_fn=tf.nn.relu,
                scope='conv1')
            feature_map = tf.reduce_mean(feature_map, [1, 2])
            feature_map = tf.expand_dims(tf.expand_dims(feature_map, 1), 2)
            logits = slim.conv2d(
                feature_map,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='logits')
            logits = tf.squeeze(logits, [1, 2], name='spatial_squeeze')
    return logits

def main(unused_argv):
    # Set the parameters here
    batch_size = 128
    num_preprocess_threads = 32
    learning_rate = 0.00005
    epochs = 1000

    # Get the list of training and validation data
    convert = False
    if "valid" in cmd_args.train_data:
        convert = True
    train_filenames, train_labels = list_images(cmd_args.train_data, convert)
    val_filenames, val_labels = list_images(cmd_args.val_data, convert)
    num_classes = len(set(train_labels))
    print("there are " + str(num_classes) + " classes")

    # Get the number of data in total for batch calculation later
    num_train_data = len(train_filenames)
    num_val_data = len(val_filenames)

    # Set up the training data pipeline
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function,
                                      num_threads=num_preprocess_threads, output_buffer_size=batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=300000)
    batched_train_dataset = train_dataset.batch(batch_size)

    # Set up the validation data pipeline
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(_parse_function,
                                  num_threads=num_preprocess_threads, output_buffer_size=batch_size)
    batched_val_dataset = val_dataset.batch(batch_size)

    # Set up the iterator
    iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                       batched_train_dataset.output_shapes)

    # Build the model
    images, labels = iterator.get_next()
    logits = build_model(images, num_classes)


    # Add loss function
    tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()

    # Get pre-trained weights
    variables_to_be_load = read_varaibles_from_file(cmd_args.variables_path)
    restore_var = [v for v in tf.global_variables() if v.name[:-2] in variables_to_be_load]
    train_variables = [v for v in tf.global_variables() if 'resnet' not in v.name]

    train_init_op = iterator.make_initializer(batched_train_dataset)
    val_init_op = iterator.make_initializer(batched_val_dataset)

    att = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    # output = att.minimize(loss, var_list=train_variables)
    output = att.minimize(loss)

    # Evaluation metrics
    prediction = tf.to_int32(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    saver2 = tf.train.Saver()

    num_batches = int(num_train_data / batch_size) + 1
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "weights_copy/trained_resnet_model_drop_1")
        print("weights loaded")
        print("there are " + str(num_batches) + " batches")
        for epoch in range(epochs):
            sess.run(train_init_op)
            print('Starting epoch %d / %d' % (epoch + 1, epochs))
            t = time.time()
            acc_at_each_epoch = []
            for batch in range(num_batches):
                # sess.run(train_init_op)
                _, acc, batch_loss = sess.run([output, accuracy, loss])
                acc_at_each_epoch.append(acc)
                #print(acc)
                if batch % 100 == 0:
                    print("At batch " + str(batch))
                    accumated_acc = sum(acc_at_each_epoch) / float(len(acc_at_each_epoch))
                    print("accuracy accumalated: " + str(accumated_acc))
                    print("loss at this batch: " + str(batch_loss))
                    elapsed = time.time() - t
                    print("it takes " + str(elapsed) + " seconds to train this 100 batches")
                    t = time.time()
                    print("==========================================================")
                    # train_acc = check_accuracy(sess, correct_prediction, train_init_op)
                    # print("training acc is: " + str(train_acc))
                    # val_acc = check_accuracy(sess, correct_prediction, val_init_op)
                    # print("validation acc is: " + str(val_acc))
                    # sess.run(train_init_op)



            print("epoch: " + str(epoch) + " is done!")
            # train_acc = check_accuracy(sess, correct_prediction, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, val_init_op)
            # print('Train accuracy: %f' % train_acc)
            print("validation acc is: " + str(val_acc) + "\n")
            saver2.save(sess, "my_weights/trained_resnet_model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
        '--config_path',
        type=str,
        default='delf_config_example.pbtxt',
        help="""
         Path to DelfConfig proto text file with configuration to be used for DELF
         extraction.
         """)
    parser.add_argument(
        '--train_data',
        type=str,
        default='/media/jason/Data1/landmarks/rawImage/train',
        help="""
         Path to train data.
         """)
    parser.add_argument(
        '--val_data',
        type=str,
        default='/media/jason/Data1/landmarks/rawImage/valid',
        help="""
             Path to validation data.
             """)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_features',
        help="""
         Directory where DELF features will be written to. Each image's features
         will be written to a file with same name, and extension replaced by .delf.
         """)
    parser.add_argument(
        '--variables_path',
        type=str,
        default='load_names.txt',
        help="""
             Variables to be loaded
             """)
    cmd_args, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

