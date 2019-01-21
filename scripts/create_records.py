#!/usr/bin/python

"""Converts VSB Power Line Faults data to TFRecords file format with Example protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

# noinspection PyPackageRequirements
import pandas as pd
# noinspection PyPackageRequirements
from pyarrow import parquet as pq
import tensorflow as tf
import numpy as np
from scipy import signal


def current_milli_time():
    return int(round(time.time() * 1000))


def read_parquet(name, cols=None):
    start_time = current_milli_time()
    print('Reading: ', name)
    parquet_file = pq.ParquetFile(name)
    table = parquet_file.read(columns=cols)
    end_time = current_milli_time()
    print('Read ', table.shape, ' cells in ', end_time - start_time, ' ms.')
    return table


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def filter_signal_low_freq(source, upper_freq, signal_duration=2.0E-2) -> (np.ndarray, np.ndarray):
    freqs = np.fft.rfft(source)
    upper_index = int(upper_freq * signal_duration)
    f_freqs = np.concatenate((np.zeros((upper_index,)),freqs[upper_index:]))
    result = np.fft.irfft(f_freqs)
    return result


def convert_to_tf(signals_parquet, metadata_csv, target_tf_prefix, batch_size):
    # read entire metadata into memory
    metadata = pd.read_csv(metadata_csv)
    # iterate over metadata in batches
    for batch_num in range(len(metadata) // batch_size + 1):
        batch_start = batch_num * batch_size
        batch_end = batch_start + batch_size
        if batch_end > len(metadata):
            batch_end = len(metadata)
        # generate column names that are included in the current batch
        cols = [str(metadata['signal_id'][i]) for i in range(batch_start, batch_end)]
        # read batch of columns from parquet file
        signals = read_parquet(signals_parquet, cols)
        # generate file name to write and open this file with TensorFlow
        tf_records_file_name = target_tf_prefix + "-" + str(batch_num) + '.tfrecords'
        print("Writing to :", tf_records_file_name)
        with tf.python_io.TFRecordWriter(tf_records_file_name) as writer:
            # iterate over metadata rows in current batch
            for index, metadata_row in metadata[batch_start:batch_end].iterrows():
                # extract signal from PyArrow Column to pandas Series using zero copy. No data transformation.
                signal_pandas = signals.column(index - batch_start).to_pandas(zero_copy_only=True)
                # get underlying NumPy array from pandas Series and it's bytes representation.
                # No data transformation happens here.
                # signal_bytes = signal_pandas.values.tobytes()

                sampling_frequency = 800000 * 100 / 2  # number of samples divided by signal duration 800000 / 20ms
                lower_frequency = 1.5E7  # cut off lower 15Mhz.
                _, _, spectrum = signal.spectrogram(signal_pandas.values, sampling_frequency)
                start_index = int(lower_frequency * spectrum.shape[0] * 2 / sampling_frequency) + 1  # x2 as one-sided
                height = spectrum.shape[0] - start_index
                width = spectrum.shape[1]
                spectrum = spectrum[start_index:][:]
                spectrum = spectrum.astype(np.float32)
                spectrum_bytes = spectrum.tobytes()
                # construct Example to store in TFRecord
                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'signal_id': _int64_feature(metadata_row[0]),
                    'id_measurement': _int64_feature(metadata_row[1]),
                    'phase': _int64_feature(metadata_row[2]),
                    # if no target given in metadata, initialize target with -1
                    'target': _int64_feature(metadata_row[3] if len(metadata_row) > 3 else -1),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'spectrum': _bytes_feature(spectrum_bytes)}))
                writer.write(tf_example.SerializeToString())
                if index % 100 == 0:
                    print('Converted ', index)


FLAGS = []


# noinspection PyUnusedLocal
def main(unused_argv):
    if not os.path.exists(FLAGS.destination_directory):
        try:
            os.makedirs(FLAGS.destination_directory)
        except OSError:
            print("Failed to create destination directory ", FLAGS.destination_directory)
            print("Using ", FLAGS.source_directory, " as destination.")
            FLAGS.destination_directory = FLAGS.source_directory
    # Convert to Examples and write the result to TFRecords.
    print('Converting train data:')
    convert_to_tf(signals_parquet=os.path.join(FLAGS.source_directory, 'train.parquet'),
                  metadata_csv=os.path.join(FLAGS.source_directory, 'metadata_train.csv'),
                  target_tf_prefix=os.path.join(FLAGS.destination_directory, 'train'),
                  batch_size=FLAGS.batch_size)
    print("Converting test data:")
    convert_to_tf(signals_parquet=os.path.join(FLAGS.source_directory, 'test.parquet'),
                  metadata_csv=os.path.join(FLAGS.source_directory, 'metadata_test.csv'),
                  target_tf_prefix=os.path.join(FLAGS.destination_directory, 'test'),
                  batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-directory',
        type=str,
        default='.',
        help='Directory with the source data: train.parquet, metadata_train.csv, test.parquet, metadata_test.csv'
    )
    parser.add_argument(
        '--destination-directory',
        type=str,
        default='.',
        help='Directory to write destination data: train-N.tfrecords and test-N.tfrecords'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2000,
        help='Number of columns of parquet file to read in memory at once.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
