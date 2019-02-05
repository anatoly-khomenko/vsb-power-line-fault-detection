import tensorflow as tf
import tensorflow_probability as tfp


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
    sample_size = tf.shape(ts)[1]
    n_dim = 160
    bucket_size = tf.cast(sample_size / n_dim, dtype=tf.int32)
    max_num = 127
    min_num = -128

    ts_std = (tf.cast(ts, dtype=tf.float32) - min_num) / (max_num - min_num)
    ts_std = ts_std * 2. - 1.

    all_stats = tf.TensorArray(size=n_dim, dtype=tf.float32,
                               element_shape=tf.TensorShape([19, None]))

    def _cond(j_v, i_v, all_stats_v):
        return tf.less(i_v, sample_size)

    def _body(j_v, i_v, all_stats_v):
        # cut each bucket to ts_range
        ts_range = ts_std[:, i_v:i_v + bucket_size]
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


def _input_fn(filename_queue, batch_size=1, take_count=None, skip_count=None, pos_weight=20.0, predict=False,
              fake=False):
    signal_feature_description = {
        'signal_id': tf.FixedLenFeature([], tf.int64),
        'id_measurement': tf.FixedLenFeature([], tf.int64),
        'phase': tf.FixedLenFeature([], tf.int64),
        'signal': tf.FixedLenFeature([], tf.string, ''),
        'spectrum': tf.FixedLenFeature([], tf.string, ''),
        'inception_v3': tf.FixedLenFeature([], tf.string, ''),
        'stats': tf.FixedLenFeature([], tf.string, ''),
        # if no target given in metadata, initialize target with -1
        'target': tf.FixedLenFeature([], tf.int64, -1)
    }

    def _parse_signal(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.parse_single_example(example_proto, signal_feature_description)
        target = parsed.pop('target')
        if fake:
            signal, weight, target = _generate_fake(predict=predict, pos_weight=pos_weight, target=target)
        else:
            # read real signal which is 800K bytes and cast it to float 32
            signal = tf.decode_raw(parsed['signal'], tf.int8)
            spectrum = tf.decode_raw(parsed['spectrum'], tf.float32)
            parsed['spectrum'] = spectrum
            inception_v3 = tf.decode_raw(parsed['inception_v3'], tf.float32)
            parsed['inception_v3'] = inception_v3
            stats = tf.decode_raw(parsed['stats'], tf.float64)
            parsed['stats'] = stats
            parsed['input_1'] = stats
            if not predict:
                weight = tf.cond(tf.math.equal(target, 1), lambda: pos_weight, lambda: 1.0)
                parsed['weight'] = weight
        parsed['signal'] = signal
        parsed.pop('signal')
        parsed.pop('spectrum')
        parsed.pop('inception_v3')
        parsed.pop('stats')
        parsed.pop('weight')
        parsed.pop('id_measurement')
        parsed.pop('phase')
        parsed.pop('signal_id')
        return parsed, target

    dataset = tf.data.TFRecordDataset(filenames=filename_queue, num_parallel_reads=8)

    if skip_count is not None:
        dataset = dataset.skip(skip_count)  # take data at the end of dataset, evaluation set
    elif take_count is not None:
        dataset = dataset.take(take_count)  # take data at the beginning of the dataset, training set

    if not predict:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))

    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_signal, batch_size=batch_size))
    dataset = dataset.prefetch(buffer_size=None)
    # dataset = dataset.map(map_func=stats_map_func, num_parallel_calls=8)
    return dataset


def get_input_fn(filename_queue, batch_size=1, take_count=None, skip_count=None, pos_weight=20.0, predict=False,
                 fake=False):
    return lambda: _input_fn(filename_queue=filename_queue, batch_size=batch_size,
                             take_count=take_count, skip_count=skip_count, pos_weight=pos_weight, predict=predict,
                             fake=fake)
