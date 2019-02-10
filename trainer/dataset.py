import tensorflow as tf
import tensorflow_probability as tfp
import multiprocessing
import os


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def dct_map_func(features, labels):
    """
    Computes DCT and repackages the result into 299x299x3 for Inception V3
    :param features: features dictionary. 'signal' key is expected as tensor of batch size x 800000
    :param labels: 0 for negative example and 1 for positive
    :return: features with added 'spectrum' key
    """
    # signal = tf.ones(shape=tf.shape(features['signal']), dtype=tf.float32)
    sb = features['signal']
    # discard last 175 values to have 799825=299*2675, reshape to 299x2675 and run DCT to get spectrogram
    sb = sb[:, :799825]
    sb = tf.cast(x=sb, dtype=tf.float32)
    sb = tf.reshape(tensor=sb, shape=[-1, 299, 2675])
    # absolute value of dct to be able to use extreme values (both maximums and negative minimums) in the below code
    sb = tf.abs(tf.spectral.dct(sb))
    # remove first harmonica, as it is the largest
    sb = sb[:, :, 1:]
    # pad 2675 to 2691 = 9 * 299
    sb = tf.pad(sb, tf.constant([[0, 0], [0, 0], [9, 8]]), mode='SYMMETRIC')
    sb = tf.reshape(tensor=sb, shape=[-1, 299, 299, 3, 3])
    # bucket padded data by calculating mean in window size 3
    sb = tf.math.reduce_mean(input_tensor=sb, axis=4, keepdims=False)
    # normalize to [0..1] range mostly
    norm = tf.math.reduce_mean(input_tensor=sb, axis=None, keep_dims=False)
    sb = tf.math.divide(sb, norm)
    features['spectrum'] = sb
    return features, labels


def stats_map_func(features, labels):
    ts = features['signal']
    sample_size = tf.shape(ts)[-1]
    n_dim = 160
    bucket_size = tf.cast(sample_size / n_dim, dtype=tf.int32)
    max_num = 127
    min_num = -128

    ts_std = (tf.cast(ts, dtype=tf.float32) - min_num) / (max_num - min_num)
    ts_std = ts_std * 2. - 1.

    all_stats = tf.TensorArray(size=n_dim, dtype=tf.float32,
                               element_shape=tf.TensorShape([19, None]))

    # noinspection PyUnusedLocal
    def _cond(j_v, i_v, all_stats_v):
        return tf.less(i_v, sample_size)

    def _body(j_v, i_v, all_stats_v):
        # cut each bucket to ts_range
        ts_range = ts_std[:, i_v:i_v + bucket_size]  # TODO: Make work also without batches ts_std - rank 0
        # calculate each feature
        mean, variance = tf.nn.moments(ts_range, axes=[1])
        std = tf.sqrt(variance)
        std_top = mean + std  # I have to test it more, but is is like a band
        std_bot = mean - std
        # note, percentiles are packed in columns, though calculated for rows
        percentil_calc = tfp.distributions.percentile(x=ts_range, q=[0., 1., 25., 50., 75., 99., 100.],
                                                      axis=1, keep_dims=False)
        max_range = percentil_calc[6] - percentil_calc[0]  # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean  # maybe it could help to understand the asymmetry
        # now, we just concatenate all the features
        stats_1 = tf.convert_to_tensor(value=[mean, std, std_top, std_bot, max_range])
        stats = tf.concat(values=[stats_1, percentil_calc, relative_percentile], axis=0)
        all_stats_v = all_stats_v.write(j_v, stats)
        j_v = tf.add(j_v, 1)
        i_v = tf.add(i_v, bucket_size)
        return j_v, i_v, all_stats_v

    i = tf.constant(0)
    j = tf.constant(0)

    j, i, all_stats = tf.while_loop(cond=_cond, body=_body, loop_vars=[j, i, all_stats], back_prop=False)

    all_stats_tensor = all_stats.concat()
    all_stats_tensor = tf.transpose(all_stats_tensor)

    features['stats_1'] = all_stats_tensor
    return features, labels


def generate_stats(stats_folder, source_dataset, prefix):
    target_tf_prefix = os.path.join(stats_folder, prefix)
    batch_num = 0
    with tf.Session() as sess:
        next_test = source_dataset.make_one_shot_iterator().get_next()
        while True:
            try:
                features, labels = sess.run(next_test)
                tf.logging.log(tf.logging.INFO, features['signal_id'])
                tf_records_file_name = target_tf_prefix + "-" + ("%05d" % batch_num) + '.tfrecords'
                batch_num += 1
                tf.logging.log(tf.logging.INFO, "Writing to :" + tf_records_file_name)
                with tf.python_io.TFRecordWriter(tf_records_file_name) as writer:
                    for i in range(len(labels)):
                        example_features = {
                            'signal_id': int64_feature(features['signal_id'][i]),
                            'id_measurement': int64_feature(features['id_measurement'][i]),
                            'phase': int64_feature(features['phase'][i]),
                            'stats_1': bytes_feature(features['stats_1'][i].tobytes()),
                        }
                        if 'target' in features:
                            example_features['target'] = int64_feature(features['target'][i])
                        example = tf.train.Example(features=tf.train.Features(feature=example_features))
                        writer.write(example.SerializeToString())
            except tf.errors.OutOfRangeError:  # end of dataset
                break


def roll_map_func(features, labels):
    ts = features['signal']
    ts_shape = tf.shape(ts)
    high = tf.cast(ts_shape[0], dtype=tf.float32)
    dist = tfp.distributions.Uniform(low=0, high=high, allow_nan_stats=False)
    shift = tf.cast(dist.sample(), tf.int32)
    ts = tf.roll(input=ts, shift=shift, axis=0)
    features['signal'] = ts
    features['id_measurement'] = features['id_measurement'] + 100000
    features['signal_id'] = features['signal_id'] + 100000
    return features, labels


def flip_map_func(features, labels):
    ts = features['signal']
    features['signal'] = tf.math.scalar_mul(-1, ts)
    features['id_measurement'] = features['id_measurement'] + 200000
    features['signal_id'] = features['signal_id'] + 200000
    return features, labels


def _generate_fake(predict, pos_weight, target):
    # create fake data which are easy to learn
    # with the same amount of positive and negative samples as in real data
    if predict:  # when predicting set target to generate different signals in the condition below
        target = tf.squeeze(tf.random.poisson(lam=0.1, shape=(1,), dtype=tf.int64, seed=12345))
    # generate fake data: if target == 1 than 1111...0000... else 0000...1111...
    weight, signal = tf.cond(tf.math.equal(target, 1),
                             lambda: (pos_weight,
                                      tf.concat([tf.ones(shape=400000, dtype=tf.float32),
                                                 tf.zeros(shape=400000, dtype=tf.float32)], 0)),
                             lambda: (1.0,
                                      tf.concat([tf.zeros(shape=400000, dtype=tf.float32),
                                                 tf.ones(shape=400000, dtype=tf.float32)], 0))
                             )
    return signal, weight, target


def load(filename_queue):
    signal_feature_description = {
        'signal_id': tf.FixedLenFeature([], tf.int64),
        'id_measurement': tf.FixedLenFeature([], tf.int64),
        'phase': tf.FixedLenFeature([], tf.int64),
        'signal': tf.FixedLenFeature([], tf.string, ''),
        'stats_1': tf.FixedLenFeature([], tf.string, ''),
        # if no target given in metadata, initialize target with -1
        'target': tf.FixedLenFeature([], tf.int64, -1)
    }
    signal_feature_encoded_types = {
        'signal': tf.int8,
        'stats_1': tf.float32
    }

    def _parse_signal(example_proto):
        parsed = tf.parse_single_example(example_proto, signal_feature_description)
        for key in signal_feature_encoded_types:
            parsed[key] = tf.decode_raw(parsed[key], signal_feature_encoded_types[key])
        return parsed, parsed['target']

    cpu_count = multiprocessing.cpu_count()
    dataset = tf.data.TFRecordDataset(filenames=filename_queue, num_parallel_reads=cpu_count)
    dataset = dataset.map(map_func=_parse_signal, num_parallel_calls=cpu_count)
    return dataset


# noinspection PyUnusedLocal
def positives(features, label):
    """
    This function only works with batches consisting of single examples
    :param features: features of the example
    :param label: label of the example
    :return: True if the example is positive
    """
    return tf.reshape(tf.equal(label, 1), [])
