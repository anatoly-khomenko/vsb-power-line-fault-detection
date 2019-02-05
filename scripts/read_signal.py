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
    c = len(batch_columns)
    fig, axs = plt.subplots(n, c)
    for i in range(n):
        if y[i]['target'] == 1:
            color = 'red'
        elif y[i]['target'] == 0:
            color = 'green'
        else:
            color = 'blue'
        col = 0
        axs[i][col].plot(y[i]['signal'], color=color)
        col += 1
        if 'spectrum' in y[i]:
            axs[i][col].matshow(y[i]['spectrum'], cmap=plt.get_cmap('cool'))
            col += 1
        if 'inception_v3' in y[i]:
            axs[i][col].plot(y[i]['inception_v3'], color=color)
            col += 1
        if 'stats' in y[i]:
            axs[i][col].plot(y[i]['stats'], color=color)
            col += 1
        if 'stats_1' in y[i]:
            axs[i][col].plot(y[i]['stats_1'], color=color)
            col += 1
    plt.ion()
    plt.show(block=True)


def plot_dataset(parsed_signal_dataset, sess, batch_size=10, total_batches=1):
    next_target = None
    # structure is "signal_id":
    # {'id_measurement': tf.int64,
    #  'phase': tf.int64,
    #  'signal': tf.int8,
    #  'spectrum': tf.float32,
    #  'inception_v3': tf.float32,
    #  'stats': tf.float32,
    #  'target': tf.int64
    # }
    signals = []
    batch_columns = set()

    iterator = parsed_signal_dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    current_batch = 0
    current_item = 0
    # Repeat total_batches times finding batch_size interchanging signal types (non-problematic, problematic).
    # Or display all records sequentially if we are in the test set
    while current_batch < total_batches:
        try:
            features, labels = sess.run(next_element)
            id_measurements = features['id_measurement']
            phases = features['phase']
            signal_ids = features['signal_id']
            signal_numpys = features['signal']
            spectrum_numpys = features['spectrum']
            inception_v3_numpys = features['inception_v3']
            stats_numpys = features['stats']
            stats_1_numpys = features['stats_1']
            for i in range(len(signal_ids)):
                if next_target is None:
                    next_target = labels[i]
                if next_target == labels[i]:
                    if labels[i] == 0:
                        next_target = 1
                    elif labels[i] == -1:
                        next_target = -1  # for test dataset where all targets are -1
                    else:
                        next_target = 0
                    signal = {
                         'id_measurement': id_measurements[i],
                         'phase': phases[i]
                    }
                    if len(signal_numpys[i]) > 0:
                        signal['signal'] = signal_numpys[i]
                        batch_columns.add('signal')
                    if len(spectrum_numpys[i]) > 0:
                        signal['spectrum'] = spectrum_numpys[i]
                        batch_columns.add('spectrum')
                    if len(inception_v3_numpys[i]) > 0:
                        signal['inception_v3'] = inception_v3_numpys[i]
                        batch_columns.add('inception_v3')
                    if len(stats_numpys[i]) > 0:
                        signal['stats'] = stats_numpys[i]
                        batch_columns.add('stats')
                    if len(stats_1_numpys[i]) > 0:
                        signal['stats_1'] = stats_1_numpys[i]
                        batch_columns.add('stats_1')
                    signal['target'] = labels[i]
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
    train_input_fn = dataset.get_input_fn(filename_queue=train_files,
                                          batch_size=FLAGS.batch_size,
                                          predict=FLAGS.predict,
                                          fake=FLAGS.fake)
    train_dataset = train_input_fn()
    # transform dataset records
    # train_dataset = train_dataset.map(map_func=dataset.stats_map_func, num_parallel_calls=8)
    end = current_milli_time()
    print('Created train dataset in ', end - start, 'ms.')
    with tf.Session() as sess:
        if FLAGS.performance:
            print('Measuring train dataset performance')
            measure_performance(parsed_signal_dataset=train_dataset, sess=sess, total_batches=FLAGS.total_batches)
        else:
            plot_dataset(train_dataset, sess, FLAGS.batch_size, FLAGS.total_batches)
    start = current_milli_time()
    test_files = glob.glob(os.path.join(FLAGS.source_directory, "test-*.tfrecords"))
    test_input_fn = dataset.get_input_fn(filename_queue=test_files,
                                         batch_size=FLAGS.batch_size,
                                         predict=FLAGS.predict,
                                         fake=FLAGS.fake)
    test_dataset = test_input_fn()
    # test_dataset = test_dataset.map(map_func=dataset.stats_map_func, num_parallel_calls=8)
    end = current_milli_time()

    print('Created test dataset in ', end - start, 'ms.')
    with tf.Session() as sess:
        if FLAGS.performance:
            print('Measuring test dataset performance')
            measure_performance(parsed_signal_dataset=test_dataset, sess=sess, total_batches=FLAGS.total_batches)
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
        '--vary',
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
