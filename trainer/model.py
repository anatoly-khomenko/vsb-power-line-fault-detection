from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def input_fn(filename_queue, batch_size=1, take_count=None, skip_count=None, pos_weight=20.0, predict=False,
             fake=False):
    signal_feature_description = {
        'spectrum': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64, 0),
    }

    def _parse_signal(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.parse_single_example(example_proto, signal_feature_description)
        # signal = tf.reshape(signal, [32, 3571])
        # signal = (tf.cast(signal, tf.float32) + 128.0)/255.0
        # target = tf.one_hot(indices=parsed['target'], depth=2)
        target = parsed['target']
        # create fake data which are easy to learn
        # with the same amount of positive and negative samples as in real data
        if fake:
            if predict:  # when predicting set target to generate different signals in the condition below
                target = tf.squeeze(
                    tf.random.poisson(lam=0.1, shape=(1,), dtype=tf.int64, seed=12345)
                )
            weight, signal = tf.cond(tf.math.equal(target, 1),
                                     lambda: (pos_weight,
                                              tf.concat([tf.ones(shape=1024, dtype=tf.float32),
                                                        tf.zeros(shape=1024, dtype=tf.float32)], 0)),
                                     lambda: (1.0,
                                              tf.concat([tf.zeros(shape=1024, dtype=tf.float32),
                                                         tf.ones(shape=1024, dtype=tf.float32)], 0))
                                     )
        else:
            signal = tf.decode_raw(parsed['spectrum'], tf.float32)
            signal.set_shape(2048)
            if predict:
                weight = 1.0
            else:
                weight = tf.cond(tf.math.equal(target, 1), lambda: pos_weight, lambda: 1.0)

        return {'signal': signal, 'weight': weight}, target

    dataset = tf.data.TFRecordDataset(filenames=filename_queue, num_parallel_reads=8)

    if skip_count is not None:
        dataset = dataset.skip(skip_count)  # take data at the end of dataset, evaluation set
    elif take_count is not None:
        dataset = dataset.take(take_count)  # take data at the beginning of the dataset, training set

    if not predict:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))

    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_signal, batch_size=batch_size))
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def get_input_fn(filename_queue, batch_size=1, take_count=None, skip_count=None, pos_weight=20.0, predict=False,
                 fake=False):
    return lambda: input_fn(filename_queue=filename_queue, batch_size=batch_size,
                            take_count=take_count, skip_count=skip_count, pos_weight=pos_weight, predict=predict,
                            fake=fake)


def linear_classifier_model_fn(features, labels, mode, params):
    # input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    input_layer = features['signal']
    # reshaped = tf.reshape(input_layer, [-1, params['feature_columns'][0].shape[0], 1])
    pool0 = tf.layers.average_pooling1d(inputs=input_layer,
                                        pool_size=100, strides=100)
    conv1 = tf.layers.conv1d(inputs=pool0,
                             filters=32, kernel_size=9,
                             padding='valid', activation=tf.nn.relu)
    conv2 = tf.layers.conv1d(inputs=conv1,
                             filters=32, kernel_size=9,
                             padding='valid', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv2,
                                    pool_size=10, strides=10)
    conv3 = tf.layers.conv1d(inputs=pool1,
                             filters=64, kernel_size=9,
                             padding='valid', activation=tf.nn.relu)
    conv4 = tf.layers.conv1d(inputs=conv3,
                             filters=64, kernel_size=9,
                             padding='valid', activation=tf.nn.relu)
    pool2 = tf.layers.average_pooling1d(inputs=conv4,
                                        pool_size=10, strides=1)
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2]])
    dropout = tf.layers.dropout(inputs=pool2_flat,
                                rate=params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout,
                             units=params['num_classes'], activation=tf.nn.softmax)

    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    tf.summary.scalar('OptimizeLoss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        label_indices = tf.argmax(input=labels, axis=1)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
                           'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(label_indices,
                                                                                         predicted_indices,
                                                                                         params['num_classes'])
                           }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def dnn_unbalanced_classifier_model_fn(features, labels, mode, params):
    input_layer = features['signal']
    hidden1 = tf.layers.dense(inputs=input_layer,
                              units=2048, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(inputs=hidden1,
                              units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=hidden2,
                             units=params['num_classes'], activation=tf.nn.softmax)

    predicted_indices = tf.argmax(input=logits, axis=1)
    target_indices = tf.argmax(input=labels, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=predicted_indices,
                                                    pos_weight=10)
    tf.summary.scalar('OptimizeLoss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        label_indices = tf.argmax(input=labels, axis=1)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
                           'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(label_indices,
                                                                                         predicted_indices,
                                                                                         params['num_classes']),
                           'precision': tf.metrics.precision(labels, probabilities),
                           'recall': tf.metrics.recall(labels, probabilities)
                           }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def cnn_model_fn(features, labels, mode, params):
    # Input Layer
    # input_layer = tf.reshape(features, [-1, features.shape[1], 1])
    input_layer = features['signal']
    conv1 = tf.layers.conv1d(inputs=input_layer,
                             filters=params['filters_1'], kernel_size=params['kernel_size_1'],
                             padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv1d(inputs=conv1,
                             filters=params['filters_2'], kernel_size=params['kernel_size_2'],
                             padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv2,
                                    pool_size=params['pool_size_1'], strides=params['strides_1'])
    conv3 = tf.layers.conv1d(inputs=pool1,
                             filters=params['filters_3'], kernel_size=params['kernel_size_3'],
                             padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv1d(inputs=conv3,
                             filters=params['filters_4'], kernel_size=params['kernel_size_4'],
                             padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.average_pooling1d(inputs=conv4,
                                        pool_size=params['pool_size_2'], strides=params['strides_2'])
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2]])
    dropout = tf.layers.dropout(inputs=pool2_flat,
                                rate=params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout,
                             units=params['num_classes'], activation=tf.nn.softmax)

    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    tf.summary.scalar('OptimizeLoss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        label_indices = tf.argmax(input=labels, axis=1)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def serving_input_fn():
    features = {'signal': tf.placeholder(dtype=tf.float32, shape=2048)}
    receiver_tensors = {'signal': tf.placeholder(dtype=tf.float32, shape=2048)}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
