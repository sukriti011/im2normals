#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path as osp
import os
import sys

from tensorflow import layers
from skimage.io import imread, imsave
import imageio

DATASET_PATH = '/home/sukritisingh/cos429challenge'
BATCH_SIZE = 20
SUMMARIES_PATH = '/home/sukritisingh/cos429challenge/summaries'
MAX_ITERATION = 150000

TRAIN_DUMP_FOLDER = '/home/sukritisingh/cos429challenge/prediction_train'
if not osp.exists(TRAIN_DUMP_FOLDER):
    os.makedirs(TRAIN_DUMP_FOLDER)

VALID_DUMP_FOLDER = '/home/sukritisingh/cos429challenge/prediction_valid'
if not osp.exists(VALID_DUMP_FOLDER):
    os.makedirs(VALID_DUMP_FOLDER)

DUMP_FOLDER = '/home/sukritisingh/cos429challenge/prediction'
if not osp.exists(DUMP_FOLDER):
    os.makedirs(DUMP_FOLDER)

def build_model():

    training = tf.placeholder(tf.bool, name='training')

    color = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
    mask = tf.placeholder(dtype=tf.bool, shape=[None, 128, 128])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

    from FCDenseNet import DenseTiramisu
    predict = DenseTiramisu(12, [4,4,4,4,4]).model(color, training)

    predict_n = tf.nn.l2_normalize(predict, axis=3)
    target_n = tf.nn.l2_normalize(target, axis=3)
    cosine_angle = tf.reduce_sum(predict_n * target_n, axis=3)
    loss = -tf.reduce_mean(tf.boolean_mask(cosine_angle, mask))

    return color, mask, target, predict_n, loss, training

def load_train_data(iteration, batch_size):
    total = 19000
    start = (iteration * batch_size) % total

    color_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)
    mask_npy = np.zeros([batch_size, 128, 128], dtype=np.uint8)
    target_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)

    for i in range(batch_size):
        color_path = osp.join(DATASET_PATH, 'train', 'color', '{}.png'.format(i + start))
        mask_path = osp.join(DATASET_PATH, 'train', 'mask', '{}.png'.format(i + start))
        target_path = osp.join(DATASET_PATH, 'train', 'normal', '{}.png'.format(i + start))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)
        target_npy[i, ...] = imread(target_path)

    color_npy = color_npy / 255.0
    target_npy = target_npy / 255.0 * 2 - 1

    return color_npy, mask_npy, target_npy

def load_valid_data(iteration, batch_size):
    total = 1000
    start = 19000 + (iteration * batch_size) % total

    color_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)
    mask_npy = np.zeros([batch_size, 128, 128], dtype=np.uint8)
    target_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)

    for i in range(batch_size):
        color_path = osp.join(DATASET_PATH, 'valid', 'color', '{}.png'.format(i + start))
        mask_path = osp.join(DATASET_PATH, 'valid', 'mask', '{}.png'.format(i + start))
        target_path = osp.join(DATASET_PATH, 'valid', 'normal', '{}.png'.format(i + start))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)
        target_npy[i, ...] = imread(target_path)

    color_npy = color_npy / 255.0
    target_npy = target_npy / 255.0 * 2 - 1

    return color_npy, mask_npy, target_npy

def load_test_data(iteration, batch_size):
    total = 2000
    start = (iteration * batch_size) % total

    color_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)
    mask_npy = np.zeros([batch_size, 128, 128], dtype=np.uint8)

    for i in range(batch_size):
        color_path = osp.join(DATASET_PATH, 'test', 'color', '{}.png'.format(i + start))
        mask_path = osp.join(DATASET_PATH, 'test', 'mask', '{}.png'.format(i + start))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)

    color_npy = color_npy / 255.0

    return color_npy, mask_npy

def scan_png_files(folder):
    '''
    folder: 1.png 3.png 4.png 6.png 7.exr unknown.mpeg
    return: ['1.png', '3.png', '4.png']
    '''
    ext = '.png'
    ret = [fname for fname in os.listdir(folder) if fname.endswith(ext)]

    return ret


def evaluate(prediction_folder, groundtruth_folder, mask_folder):
    '''
    Evaluate mean angle error of predictions in the prediction folder,
    given the groundtruth and mask images.
    '''
    # Scan folders to obtain png files
    if mask_folder is None:
        mask_folder = os.path.join(groundtruth_folder, '..', 'mask')

    pred_pngs = scan_png_files(prediction_folder)
    gt_pngs = scan_png_files(groundtruth_folder)
    mask_pngs = scan_png_files(mask_folder)

    pred_diff_gt = set(pred_pngs).difference(gt_pngs)
    #assert len(pred_diff_gt) == 0, \
    #    'No corresponding groundtruth file for the following files:\n' + '\n'.join(pred_diff_gt)
    pred_diff_mask = set(pred_pngs).difference(mask_pngs)
    #assert len(pred_diff_mask) == 0, \
    #    'No corresponding mask file for the following files:\n' + '\n'.join(pred_diff_mask)

    # Measure: mean angle error over all pixels
    mean_angle_error = 0
    total_pixels = 0
    for fname in pred_pngs:
        prediction = imageio.imread(os.path.join(prediction_folder, fname))
        groundtruth = imageio.imread(os.path.join(groundtruth_folder, fname))
        mask = imageio.imread(os.path.join(mask_folder, fname)) # Greyscale image

        prediction = ((prediction / 255.0) - 0.5) * 2
        groundtruth = ((groundtruth / 255.0) - 0.5) * 2

        total_pixels += np.count_nonzero(mask)
        mask = mask != 0

        a11 = np.sum(prediction * prediction, axis=2)[mask]
        a22 = np.sum(groundtruth * groundtruth, axis=2)[mask]
        a12 = np.sum(prediction * groundtruth, axis=2)[mask]

        cos_dist = a12 / np.sqrt(a11 * a22)
        cos_dist[np.isnan(cos_dist)] = -1
        cos_dist = np.clip(cos_dist, -1, 1)
        angle_error = np.arccos(cos_dist)
        mean_angle_error += np.sum(angle_error)

    return mean_angle_error / total_pixels

def train():
    color, mask, target, predict_n, loss, training = build_model()
    loss_summ = tf.summary.scalar('training_loss', loss)
    writer = tf.summary.FileWriter(SUMMARIES_PATH)

    learning_rate = 1e-4
    learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
    # train_op = optimizer.minimize(loss)
    original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()

    print('train phase')
    with tf.device("/gpu:0"):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(MAX_ITERATION+1):
                color_npy, mask_npy, target_npy = load_train_data(i, BATCH_SIZE)
                feed_dict = {
                    color: color_npy,
                    mask: mask_npy,
                    target: target_npy,
                    learning_rate_placeholder: learning_rate,
                    training: True
                }
                loss_val, summ, _ = sess.run([loss, loss_summ, train_op], feed_dict=feed_dict)
                writer.add_summary(summ, i)
                sys.stdout.write('\r%d %f' % (i, loss_val))
                sys.stdout.flush()

                if i % 100 == 0:
                    sys.stdout.write('\ntrain ')
                    total_loss = 0
                    for j in range(16):
                       color_npy, mask_npy, target_npy = load_train_data(j, 1)
                       feed_dict = {
                           color: color_npy,
                           mask: mask_npy,
                           target: target_npy,
                           learning_rate_placeholder: learning_rate,
                           training: False
                       }
                       predict_val, loss_val = sess.run([predict_n, loss], feed_dict=feed_dict)
                       total_loss = total_loss + 1 * loss_val / 16
                       sys.stdout.write('\rtrain %f (%d)' % (loss_val, j))
                       predict_img = ((predict_val.squeeze(0) + 1) / 2 * 255).astype(np.uint8)
                       imsave(osp.join(TRAIN_DUMP_FOLDER, '{}.png'.format(j)), predict_img)
                    sys.stdout.write('\rtrain %f\n' % total_loss)
                    mae = evaluate(TRAIN_DUMP_FOLDER, osp.join(DATASET_PATH, 'train', 'normal'), osp.join(DATASET_PATH, 'train', 'mask'))
                    sys.stdout.write('train mae %f\n' % mae)

                if i > 499 and i % 500 == 0:
                    sys.stdout.write('\nvalid ')
                    total_loss = 0
                    for j in range(1000):
                       color_npy, mask_npy, target_npy = load_valid_data(j, 1)
                       feed_dict = {
                           color: color_npy,
                           mask: mask_npy,
                           target: target_npy,
                           learning_rate_placeholder: learning_rate,
                           training: False
                       }
                       predict_val, loss_val = sess.run([predict_n, loss], feed_dict=feed_dict)
                       total_loss = total_loss + 1 * loss_val / 1000
                       sys.stdout.write('\rvalid %f (%d)' % (loss_val, j))
                       predict_img = ((predict_val.squeeze(0) + 1) / 2 * 255).astype(np.uint8)
                       imsave(osp.join(VALID_DUMP_FOLDER, '{}.png'.format(19000+j)), predict_img)
                    sys.stdout.write('\rvalid %f\n' % total_loss)
                    mae = evaluate(VALID_DUMP_FOLDER, osp.join(DATASET_PATH, 'valid', 'normal'), osp.join(DATASET_PATH, 'valid', 'mask'))
                    sys.stdout.write('valid mae %f\n\n' % mae)
                    save_path = saver.save(sess,  os.path.join('models', 'model_%d.tf'%(i)))
                    sys.stdout.write('Model saved in file: %s\n' % save_path)

                if i > 999 and i % 1000 == 0:
                    # run on test data and save in predictions folder
                    pred_folder = '{}_{}'.format(DUMP_FOLDER, i)
                    os.makedirs(pred_folder, exist_ok=True)
                    sys.stdout.write('saving predictions to %s\n' % pred_folder)
                    for j in range(2000):
                       color_npy, mask_npy = load_test_data(j, 1)
                       feed_dict = {
                           color: color_npy,
                           mask: mask_npy,
                           training: False
                       }
                       predict_val, = sess.run([predict_n], feed_dict=feed_dict)
                       predict_img = ((predict_val.squeeze(0) + 1) / 2 * 255).astype(np.uint8)
                       sys.stdout.write('\rtest (%d)' % (j))
                       imsave(osp.join(pred_folder, '{}.png'.format(j)), predict_img)
                    sys.stdout.write('\n')

if __name__ == '__main__':
    train()
