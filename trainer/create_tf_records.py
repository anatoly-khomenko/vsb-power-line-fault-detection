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
            # construct Example to store in TFRecord
            features_dict = {
                'signal_id': dataset.int64_feature(signal_id),
                'id_measurement': dataset.int64_feature(id_measurement),
                'phase': dataset.int64_feature(phase),
                'signal': dataset.bytes_feature(signal_bytes),
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
