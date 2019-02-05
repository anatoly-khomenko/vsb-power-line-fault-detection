import glob
import os
import argparse
import sys

import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyPackageRequirements
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from trainer import dataset


def process_files(suffix, input_dir, log_dir, output_dir, features_tensor, input_tensor, batch_size):
    log_dir = log_dir + suffix
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    train_files = glob.glob(os.path.join(input_dir, suffix + '-*.tfrecords'))
    # train_files = [os.path.join(input_dir, suffix + '-0.tfrecords')]
    train_input_fn = dataset.get_input_fn(filename_queue=train_files, batch_size=batch_size, predict=True)
    train_dataset = train_input_fn()
    train_dataset = train_dataset.map(map_func=dataset.dct_map_func, num_parallel_calls=8)
    tf_records_file_name = os.path.join(output_dir, suffix + '-0.tfrecords')
    with tf.Session() as sess, tf.python_io.TFRecordWriter(tf_records_file_name) as writer:
        init = tf.global_variables_initializer()
        sess.run(init)
        iterator = train_dataset.make_one_shot_iterator()

        labels = []
        label_indices = None
        all_features = None
        try:
            for i in range(700):
                spectrum = sess.run(iterator.get_next())
                features = sess.run(features_tensor, feed_dict={input_tensor: spectrum[0]['spectrum']})
                if all_features is None:
                    all_features = features
                else:
                    all_features = np.append(all_features, features, 0)
                if label_indices is None:
                    label_indices = spectrum[1]
                else:
                    label_indices = np.append(label_indices, spectrum[1], 0)
                one_hot_label = sess.run(tf.one_hot(indices=spectrum[1] + 1, depth=3))
                labels.append(one_hot_label[0])
                for j in range(len(features)):
                    features_dict = {
                        'signal_id': dataset.int64_feature(spectrum[0]['signal_id'][j]),
                        'id_measurement': dataset.int64_feature(spectrum[0]['id_measurement'][j]),
                        'phase': dataset.int64_feature(spectrum[0]['phase'][j]),
                        'target': dataset.int64_feature(spectrum[1][j]),
                        'inception_v3': dataset.bytes_feature(features[j].astype(np.float32).tobytes())}
                    tf_example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                    writer.write(tf_example.SerializeToString())
                writer.flush()
                print('Processed ' + suffix + ' ', i)
        except tf.errors.OutOfRangeError:
            print('Finished dataset: ' + suffix)

        saver_features = tf.Variable(all_features)
        saver = tf.train.Saver([saver_features])
        sess.run(saver_features.initializer)

        metadata = os.path.join(log_dir, 'metadata.tsv')

        with open(metadata, 'w') as metadata_file:
            for row in label_indices:
                metadata_file.write('%d\n' % row)

        saver.save(sess, os.path.join(log_dir, 'signals.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = saver_features.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata.tsv'
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)

        # reader.plot_all(all_features, labels)


FLAGS = []


# noinspection PyUnusedLocal
def main(unused_argv):
    module_spec_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    module_spec = hub.load_module_spec(module_spec_url)

    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        module = hub.Module(module_spec)
        features_tensor = module(input_tensor)
        # process_files('train', FLAGS.input_dir, FLAGS.log_dir, FLAGS.output_dir, features_tensor, input_tensor,
        #               FLAGS.batch_size)
        process_files('test', FLAGS.input_dir, FLAGS.log_dir, FLAGS.output_dir, features_tensor, input_tensor,
                      FLAGS.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='../../tf_original',
        help='Directory to read source data: train-N.tfrecords and test-N.tfrecords'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../../tf_inception_v3',
        help='Directory to write destination data: train-N.tfrecords and test-N.tfrecords'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../../hub_model_output_',
        help='Directory to write data for Projector plugin of TensorBoard. Will have "test" and "train" appended'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=30,
        help='Number of columns of parquet file to read in memory at once.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
