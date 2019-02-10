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
# noinspection PyPackageRequirements
import numpy as np
from trainer import dataset


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


def plot_all(y, batch_columns):
    n = len(y)
    c = len(batch_columns)+1
    fig, axs = plt.subplots(n, c)
    for i in range(n):
        if y[i]['target'] == 1:
            color = 'red'
        elif y[i]['target'] == 0:
            color = 'green'
        else:
            color = 'blue'
        col = 0
        for column in batch_columns:
            if len(y[i][column].shape) == 2:
                axs[i][col].matshow(y[i][column], cmap=plt.get_cmap('cool'))
            elif len(y[i][column].shape) == 3:
                axs[i][col].imshow(y[i][column], cmap=plt.get_cmap('cool'))
            else:
                axs[i][col].plot(y[i][column], color=color)
            axs[i][col].title.set_text(column)
            col += 1
    plt.ion()
    plt.show(block=True)


def plot_dataset(parsed_signal_dataset, sess, batch_size=10, total_batches=1, alternate=True):
    next_target = None
    # signals structure is [
    # {'signal_id': tf.int64,
    #  'id_measurement': tf.int64,
    #  'phase': tf.int64,
    #  'signal': tf.int8,
    #  'spectrum': tf.float32,
    #  'inception_v3': tf.float32,
    #  'stats': tf.float32,
    #  'target': tf.int64
    # }]
    signals = []
    batch_columns = []
    columns = ['signal', 'spectrum', 'inception_v3', 'stats', 'stats_1', 'rolled', 'scaled']

    iterator = parsed_signal_dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    current_batch = 0
    current_item = 0
    # Repeat total_batches times finding batch_size interchanging signal types (non-problematic, problematic).
    # Or display all records sequentially if we are in the test set
    while current_batch < total_batches:
        try:
            features, labels = sess.run(next_element)
            if not isinstance(labels, list):
                labels = [labels]
                new_features = {}
                for key in features:
                    new_features[key] = [features[key]]
                features = new_features
            for i in range(len(labels)):
                if next_target is None:
                    next_target = labels[i]
                if not alternate or next_target == labels[i]:
                    if labels[i] == 0:
                        next_target = 1
                    elif labels[i] == -1:
                        next_target = -1  # for test dataset where all targets are -1
                    else:
                        next_target = 0
                    signal = {
                         'signal_id': features['signal_id'][i],
                         'id_measurement': features['id_measurement'][i],
                         'phase': features['phase'][i],
                         'target': labels[i]
                    }
                    for column in columns:
                        if column in features and len(features[column][i]) > 0:
                            signal[column] = features[column][i]
                            if column not in batch_columns:
                                batch_columns.append(column)

                    signals.append(signal)

                    current_item = current_item + 1
                    if current_item == batch_size:
                        plot_all(signals, batch_columns)
                        signals = []
                        current_batch += 1
                        current_item = 0
        except tf.errors.OutOfRangeError as e:
            print(e)
            break


def measure_performance(parsed_signal_dataset, sess, total_batches=1):
    current_batch = 0
    total_records_processed = 0
    avg = 0.0
    start = current_milli_time()
    iterator = parsed_signal_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    while current_batch < total_batches:
        try:
            signal_features_batch = sess.run(next_element)
            signal_numpy_batch = signal_features_batch[0]['signal']
            # noinspection PyUnusedLocal
            signal_label_batch = signal_features_batch[1]
            for i in range(len(signal_numpy_batch)):
                avg += signal_numpy_batch[i]
                total_records_processed += 1
            current_batch += 1
        except tf.errors.OutOfRangeError:
            break
    end = current_milli_time()
    avg = avg / total_records_processed
    print('Read ', total_records_processed, ' records in ', end - start, 'ms')
    print('Average of signal elements is ', avg)


FLAGS = []


# noinspection PyUnusedLocal
def main(unused_argv):
    start = current_milli_time()
    train_files = glob.glob(os.path.join(FLAGS.source_directory, "train-*.tfrecords"))
    # train_input_fn = dataset.get_input_fn(filename_queue=train_files,
    #                                       batch_size=FLAGS.batch_size,
    #                                       predict=FLAGS.predict,
    #                                       fake=FLAGS.fake)
    # train_dataset = train_input_fn()
    train_dataset = dataset.load(filename_queue=train_files)
    # transform dataset records
    # train_dataset = train_dataset.map(map_func=dataset.stats_map_func, num_parallel_calls=8)
    end = current_milli_time()
    print('Created train dataset in ', end - start, 'ms.')
    with tf.Session() as sess:
        if FLAGS.performance:
            print('Measuring train dataset performance')
            measure_performance(parsed_signal_dataset=train_dataset, sess=sess, total_batches=FLAGS.total_batches)
        else:
            plot_dataset(train_dataset, sess, FLAGS.batch_size, FLAGS.total_batches, FLAGS.alternate)
    start = current_milli_time()
    test_files = glob.glob(os.path.join(FLAGS.source_directory, "test-*.tfrecords"))
    # test_input_fn = dataset.get_input_fn(filename_queue=test_files,
    #                                      batch_size=FLAGS.batch_size,
    #                                      predict=FLAGS.predict,
    #                                      fake=FLAGS.fake)
    # test_dataset = test_input_fn()
    # test_dataset = test_dataset.map(map_func=dataset.stats_map_func, num_parallel_calls=8)
    test_dataset = dataset.load(filename_queue=test_files)
    end = current_milli_time()

    print('Created test dataset in ', end - start, 'ms.')
    with tf.Session() as sess:
        if FLAGS.performance:
            print('Measuring test dataset performance')
            measure_performance(parsed_signal_dataset=test_dataset, sess=sess, total_batches=FLAGS.total_batches)
        else:
            plot_dataset(test_dataset, sess, FLAGS.batch_size, FLAGS.total_batches, FLAGS.alternate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-directory',
        type=str,
        default='.',
        help='Directory to write destination data: train-N.tfrecords and test-N.tfrecords'
    )
    parser.add_argument(
        '--performance',
        action="store_true",
        help='Measure reading data performance without actually displaying data'
    )
    parser.add_argument(
        '--fake',
        action="store_true",
        help='Generate fake signals (data are read from disk anyways)'
    )
    parser.add_argument(
        '--alternate',
        action="store_true",
        help='Take one negative than one positive sample when plotting'
    )
    parser.add_argument(
        '--predict',
        action="store_true",
        help='Do not shuffle records when reading'
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
