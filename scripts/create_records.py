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
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from trainer import dataset


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


def min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def calc_stats(ts, n_dim=160, min_max=(-1, 1)):
    max_num = 127
    min_num = -128
    sample_size = 800000
    # convert data into -1 to 1
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for interact any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std()  # standard deviation
        std_top = mean + std  # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribution analysis from each chunk
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]  # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean  # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        new_ts.append(np.concatenate(
            [np.asarray([mean, std, std_top, std_bot, max_range]), percentil_calc, relative_percentile]))
    return np.asarray(new_ts)


def _create_examples(id_measurement, metadata, signals_parq):
    examples = []
    for phase in [0, 1, 2]:
        try:
            record = metadata.loc[(id_measurement, phase)]
            signal_id = record.loc['signal_id']
            if 'target' in record:
                target = record.loc['target']
            else:
                target = None
            # extract signal from PyArrow Column to pandas Series using zero copy. No data transformation.
            signal_pandas = signals_parq.column(str(signal_id)).to_pandas(zero_copy_only=True)
            # get underlying NumPy array from pandas Series and it's bytes representation.
            # No data transformation happens here.
            signal_bytes = signal_pandas.values.tobytes()
            # Add statistics data
            stats = calc_stats(signal_pandas)
            # construct Example to store in TFRecord
            features_dict = {
                'signal_id': dataset.int64_feature(signal_id),
                'id_measurement': dataset.int64_feature(id_measurement),
                'phase': dataset.int64_feature(phase),
                'signal': dataset.bytes_feature(signal_bytes),
                'stats':  dataset.bytes_feature(stats.tobytes())
            }
            # no target given in test metadata, in this case no target feature will be present
            if target is not None:
                features_dict['target'] = dataset.int64_feature(target)
            examples.append(tf.train.Example(features=tf.train.Features(feature=features_dict)))
        except KeyError as e:
            # id_measurement or phase is not present
            print("Warning: ", e)
            pass
    return examples


def convert_to_tf(signals_file_name, metadata_file_name, target_tf_prefix, batch_size):
    # read entire metadata into memory
    metadata = pd.read_csv(metadata_file_name)
    metadata.set_index(keys=['id_measurement', 'phase'],
                       drop=False, inplace=True, verify_integrity=True)
    metadata.sort_index(level='id_measurement', inplace=True)
    metadata_id_measurements = metadata.index.get_level_values('id_measurement').unique()
    n_measurements = len(metadata_id_measurements)
    num_batches = n_measurements // batch_size + 1
    # iterate over metadata in batches
    for batch_num in range(num_batches):
        print("Batch#", batch_num, ' of ', num_batches)
        batch_start = batch_num * batch_size
        batch_end = batch_start + batch_size
        if batch_end > n_measurements:
            batch_end = n_measurements
        metadata_id_measurements_batch = metadata_id_measurements[batch_start:batch_end]
        metadata_batch = metadata.loc[metadata_id_measurements_batch]
        # get column names that are included in the current batch
        cols = map(str, list(metadata_batch['signal_id']))
        # read batch of columns from parquet file
        signals_parq = read_parquet(signals_file_name, cols)
        # generate file name to write and open this file with TensorFlow
        tf_records_file_name = target_tf_prefix + "-" + ("%05d" % batch_num) + '.tfrecords'
        print("Writing to :", tf_records_file_name)
        with tf.python_io.TFRecordWriter(tf_records_file_name) as writer:
            # iterate over metadata rows in current batch
            for id_measurement in tqdm(metadata_id_measurements_batch):
                examples = _create_examples(id_measurement, metadata_batch, signals_parq)
                for example in examples:
                    writer.write(example.SerializeToString())


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
    convert_to_tf(signals_file_name=os.path.join(FLAGS.source_directory, 'train.parquet'),
                  metadata_file_name=os.path.join(FLAGS.source_directory, 'metadata_train.csv'),
                  target_tf_prefix=os.path.join(FLAGS.destination_directory, 'train'),
                  batch_size=FLAGS.batch_size)
    print("Converting test data:")
    convert_to_tf(signals_file_name=os.path.join(FLAGS.source_directory, 'test.parquet'),
                  metadata_file_name=os.path.join(FLAGS.source_directory, 'metadata_test.csv'),
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
        default=200,
        help='Number of columns of parquet file to read in memory at once.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
