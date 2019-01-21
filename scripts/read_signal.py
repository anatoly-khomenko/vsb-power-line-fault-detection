#!/usr/bin/python

"""
Reads VSB Power Line Faults data converted to TFRecords file format with Example protocol.
Depending on parameters given, either outputs plots or measures performance of the system.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import glob

# noinspection PyPackageRequirements
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def current_milli_time():
    return int(round(time.time() * 1000))


def filter_signal(source, lower_freq, upper_freq, signal_duration=2.0E-2) -> (np.ndarray, np.ndarray):
    freqs = np.fft.rfft(source)
    upper_index = upper_freq * signal_duration
    lower_index = lower_freq * signal_duration
    f_freqs = freqs.copy()
    for i in range(int(lower_index), int(upper_index) + 1):
        f_freqs[i] = 0
    result = np.fft.irfft(f_freqs)
    return f_freqs, result


def read_tf(target_tf):
    raw_signal_dataset = tf.data.TFRecordDataset(target_tf)
    signal_feature_description = {
        # 'width': tf.FixedLenFeature([], tf.int64),
        # 'height': tf.FixedLenFeature([], tf.int64),
        'spectrum': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
       }

    def _parse_signal_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.parse_single_example(example_proto, signal_feature_description)
        # width = parsed['width']
        # height = parsed['height']
        spectrum = tf.decode_raw(parsed['spectrum'], tf.float32)
        # spectrum = tf.reshape(spectrum, [32, 3571])
        spectrum.set_shape(2048)
        spectrum = tf.cast(spectrum, tf.float32)
        target = tf.one_hot(indices=parsed['target'] + 1, depth=3)
        # target = parsed['target']
        return spectrum, target

    dataset = raw_signal_dataset.map(_parse_signal_function)

    return dataset


def plot_all(y, labels, size=(13, 4), ylim=None, start=0, end=-1):
    n = len(y)
    fig, axs = plt.subplots(n, 1, sharex='all')
    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(size[0], size[1] * n)
    if end == -1:
        end = len(y[0])
    for i in range(n):
        if labels[i][0] == 0 and labels[i][1] == 0 and labels[i][2] == 1:
            color = 'red'
        elif labels[i][0] == 0 and labels[i][1] == 1 and labels[i][2] == 0:
            color = 'green'
        else:
            color = 'blue'
        if ylim is not None:
            axs[i].set_ylim(ylim)
        if len(y[i].shape) > 1 and y[i].shape[0] > 1:
            t = np.arange(0, 2.0E-2, 2.0E-2/y[i].shape[1])
            f = np.arange(1.5E7, 4.0E7, (4.0E7 - 1.5E7)/y[i].shape[0])
            axs[i].pcolormesh(t, f, y[i], cmap=plt.get_cmap('prism'))
        else:
            axs[i].plot(y[i][start:end], color=color)
    plt.ion()
    plt.show(block=True)


def plot_dataset(parsed_signal_dataset, sess, batch_size=10, total_batches=1):
    next_target = None
    signals = []
    labels = []
    iterator = parsed_signal_dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    current_batch = 0
    # Repeat 100 times finding 10 interchanging signal types (non-problematic, problematic).
    # Or display all records sequentially if we are in the test set
    while current_batch < total_batches:
        try:
            signal_features = sess.run(next_element)
            signal_numpy = signal_features[0]
            signal_label = signal_features[1]
            if next_target is None:
                next_target = signal_label
            if next_target[0] == signal_label[0] \
                    and next_target[1] == signal_label[1] \
                    and next_target[2] == signal_label[2]:
                if signal_label[0] == 0 and signal_label[1] == 1 and signal_label[2] == 0:
                    next_target = [0., 0., 1.]
                elif signal_label[0] == 0 and signal_label[1] == 0 and signal_label[2] == 1:
                    next_target = [0., 1., 0.]
                else:
                    next_target = [1., 0., 0.]
                signals.append(signal_numpy)
                labels.append(signal_label)
                if len(signals) == batch_size:
                    plot_all(signals, labels)
                    labels = []
                    signals = []
                    current_batch += 1
        except tf.errors.OutOfRangeError:
            break


def measure_performance(parsed_signal_dataset, sess, batch_size=10, total_batches=1, shuffle=False, buffer_size=1000):
    current_batch = 0
    total_records_processed = 0
    avg = 0.0
    start = current_milli_time()
    if shuffle:
        parsed_signal_dataset = parsed_signal_dataset.shuffle(buffer_size)
    iterator = parsed_signal_dataset.batch(batch_size).make_one_shot_iterator()
    next_element = iterator.get_next()
    while current_batch < total_batches:
        try:
            signal_features_batch = sess.run(next_element)
            signal_numpy_batch = signal_features_batch[0]
            # noinspection PyUnusedLocal
            signal_label_batch = signal_features_batch[1]
            for i in range(len(signal_numpy_batch)):
                avg += signal_numpy_batch[i][0]
                total_records_processed += 1
            current_batch += 1
        except tf.errors.OutOfRangeError:
            break
    end = current_milli_time()
    avg = avg / total_records_processed
    print('Read ', total_records_processed, ' records in ', end - start, 'ms')
    print('Average of signal element 0 is ', avg)


FLAGS = []


# noinspection PyUnusedLocal
def main(unused_argv):
    start = current_milli_time()
    train_files = glob.glob(os.path.join(FLAGS.source_directory, "Inception-V3-features-train.tfrecords"))
    train_dataset = read_tf(train_files)
    end = current_milli_time()
    print('Created train dataset in ', end - start, 'ms.')
    with tf.Session() as sess:
        if FLAGS.performance:
            print('Measuring train dataset performance')
            measure_performance(train_dataset, sess, FLAGS.batch_size, FLAGS.total_batches, FLAGS.shuffle)
        else:
            plot_dataset(train_dataset, sess, FLAGS.batch_size, FLAGS.total_batches)
    start = current_milli_time()
    test_files = glob.glob(os.path.join(FLAGS.source_directory, "test-*.tfrecords"))
    test_dataset = read_tf(test_files)
    end = current_milli_time()
    print('Created test dataset in ', end - start, 'ms.')
    with tf.Session() as sess:
        if FLAGS.performance:
            print('Measuring test dataset performance')
            measure_performance(test_dataset, sess, FLAGS.batch_size, FLAGS.total_batches, FLAGS.shuffle)
        else:
            plot_dataset(test_dataset, sess, FLAGS.batch_size, FLAGS.total_batches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-directory',
        type=str,
        default='.',
        help='Directory to write destination data: train-N.tfrecords and test-N.tfrecords'
    )
    parser.add_argument(
        '-p', '--performance',
        action="store_true",
        help='Measure reading data performance without actually displaying data'
    )
    parser.add_argument(
        '-s', '--shuffle',
        action="store_true",
        help='Shuffle records when reading'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of columns of parquet file to read in memory at once.'
    )
    parser.add_argument(
        '--total-batches',
        type=int,
        default=1,
        help='Number of times to display plots.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
