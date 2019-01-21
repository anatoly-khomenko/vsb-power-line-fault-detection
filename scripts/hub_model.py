import glob
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def input_fn(filename_queue, batch_size=1, predict=False):
    dataset = tf.data.TFRecordDataset(filenames=filename_queue, num_parallel_reads=8)
    signal_feature_description = {
        'spectrum': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64, default_value=-1),
    }

    def _parse_signal(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.parse_single_example(example_proto, signal_feature_description)
        signal = tf.decode_raw(parsed['spectrum'], tf.float32)
        # signal.set_shape(2048)
        signal = tf.reshape(signal, [32, 3571])
        # signal = (tf.cast(signal, tf.float32) + 128.0)/255.0
        # target = tf.one_hot(indices=parsed['target'], depth=2)
        target = parsed['target']
        return {'signal': signal}, target

    if not predict:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_parse_signal, batch_size=batch_size))
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


LOG_DIR = '../../hub_model_output_test'
os.makedirs(LOG_DIR, exist_ok=True)
module_spec_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
module_spec = hub.load_module_spec(module_spec_url)

height, width = hub.get_expected_image_size(module_spec)
with tf.Graph().as_default() as graph:
    input_tensor = tf.placeholder(tf.float32, [None, 32, 3571])
    padded = tf.pad(input_tensor, [[0, 0], [0, 0], [0, 17]])
    reshaped = tf.reshape(padded, [-1, 128, 299, 3])
    padded_1 = tf.pad(reshaped, [[0, 0], [0, 171], [0, 0], [0, 0]])
    module = hub.Module(module_spec)
    features_tensor = module(padded_1)

    train_files = glob.glob(os.path.join("../../tf_spectrum_filtered_15MHz", "test-*.tfrecords"))
    # train_files = [os.path.join("../../tf_spectrum_filtered_15MHz", "train-0.tfrecords")]
    train_dataset = input_fn(train_files, 30, True)
    tf_records_file_name = os.path.join("../../tf_spectrum_filtered_15MHz", 'Inception-V3-features-test.tfrecords')
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
                features = sess.run(features_tensor, feed_dict={input_tensor: spectrum[0]['signal']})
                if all_features is None:
                    all_features = features
                else:
                    all_features = np.append(all_features, features, 0)
                if label_indices is None:
                    label_indices = spectrum[1]
                else:
                    label_indices = np.append(label_indices, spectrum[1], 0)
                one_hot_label = sess.run(tf.one_hot(indices=spectrum[1]+1, depth=3))
                labels.append(one_hot_label[0])
                for j in range(len(features)):
                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        'target': _int64_feature(spectrum[1][j]),
                        'spectrum': _bytes_feature(features[j].astype(np.float32).tobytes())}))
                    writer.write(tf_example.SerializeToString())
                writer.flush()
                print("Processed ", i)
        except tf.errors.OutOfRangeError:
            print("Finished dataset.")

        saver_features = tf.Variable(all_features)
        saver = tf.train.Saver([saver_features])
        sess.run(saver_features.initializer)

        metadata = os.path.join(LOG_DIR, 'metadata.tsv')

        with open(metadata, 'w') as metadata_file:
            for row in label_indices:
                metadata_file.write('%d\n' % row)

        saver.save(sess, os.path.join(LOG_DIR, 'signals.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = saver_features.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = "metadata.tsv"
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

        # reader.plot_all(all_features, labels)
