from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
from delf import delf_v1

from tensorflow.python.platform import app
from delf import feature_io
from sklearn.neighbors import BallTree
import time
from nets import resnet_v1
import random

slim = tf.contrib.slim


class Siamese:
    # Create model
    def __init__(self, images1, images2, similarity, is_training=True):
        with tf.variable_scope("siamese") as scope:
            self.n1 = self.network(images1, is_training=is_training)
            scope.reuse_variables()
            self.n2 = self.network(images2, is_training=is_training, reuse=True)
        self.similarity = similarity
        self.loss = self.loss_with_spring()

    def network(self, images, is_training=True, reuse=None):
        model = delf_v1.DelfV1()
        net, end_points = model.GetResnet50Subnetwork(images, global_pool=True, is_training=is_training, reuse=reuse)

        with slim.arg_scope(
                resnet_v1.resnet_arg_scope(
                    weight_decay=0.0001, batch_norm_scale=True)):
            with slim.arg_scope([slim.batch_norm], is_training=True):

                feature_map = end_points['siamese/resnet_v1_50/block3']
                feature_map = slim.conv2d(
                    feature_map,
                    512,
                    1,
                    rate=1,
                    activation_fn=tf.nn.relu,
                    scope='conv1')
                feature_map = tf.reduce_mean(feature_map, [1, 2])
                feature_map = tf.expand_dims(tf.expand_dims(feature_map, 1), 2)
        return feature_map

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.similarity
        labels_f = tf.subtract(1.0, self.similarity, name="1-yi")  # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.n1, self.n2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")

        return loss


def load_image():
    pass


def create_hash_for_positive(class_indices):
    d = {}
    for i in range(len(class_indices)):
        if i not in d:
            d[i] = []

        # search backward
        for j in range(i):
            if class_indices[i - j - 1] == class_indices[i]:
                d[i].append(i-j-1)
            else:
                break

        # search forward
        for j in range(len(class_indices) - i - 1):
            if class_indices[i + 1 + j] == class_indices[i]:
                d[i].append(i + 1 + j)
            else:
                break

    return d


def sample(positive_hash):
    # sample the positive
    num_imgs = len(positive_hash.keys())
    positive_imgs1 = positive_hash.keys()
    positive_imgs2 = []
    for key in positive_imgs1:
        imgs = positive_hash[key]
        index = random.randint(0, len(imgs) - 1)
        print(imgs)
        print(index)
        positive_imgs2.append(imgs[index])

    negative_imgs1 = positive_hash.keys()
    negative_imgs2 = []
    for key in negative_imgs1:
        neg_class = random.randint(0, len(negative_imgs1))
        while neg_class == key:
            neg_class = random.randint(0, len(negative_imgs1) - 1)
        imgs = positive_hash[neg_class]
        index = random.randint(0, len(imgs) - 1)
        negative_imgs2.append(imgs[index])

    index1 = positive_imgs1 + negative_imgs1
    index2 = positive_imgs2 + negative_imgs2

    similarity = [1] * num_imgs + [0] * num_imgs
    return index1, index2, similarity


def main():
    # load the images into the memory
    # assuming that the images has dimension [num_imgs, 224, 224, 3]
    images, class_indices = load_image()

    # create the hash for the positive samples
    positive_hash = create_hash_for_positive(class_indices)

    # sample from the images
    index1, index2, similarity = sample(positive_hash)

    # maybe do some shuffle here?

    # grab images from these indexes and pass it to the placeholder?
    imgs_1 = images[index1]
    imgs_2 = images[index2]


    img1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    img2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    sim = tf.placeholder(tf.float32, [None])
    net = Siamese(img1, img2, sim)
    print(net.n1)
    print(net.n2)

    feed_dict = {img1: imgs_1, img2: imgs_2, sim: similarity}


# if __name__ == "__main__":
#     main()
#
