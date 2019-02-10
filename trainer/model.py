from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MetadataProfilerHook(tf.train.SessionRunHook):
    def __init__(self,
                 save_steps=None,
                 save_secs=None,
                 output_dir=""):
        self._file_writer = None
        self._next_step = None
        self._global_step_tensor = None
        self._request_summary = None
        self._output_dir = output_dir
        self._timer = tf.train.SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = tf.train.get_global_step()
        self._file_writer = tf.summary.FileWriterCache.get(self._output_dir)
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use MetadataProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
                self._next_step is not None and
                self._timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                if self._request_summary else None)

        return tf.train.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        if self._next_step is None:
            # Update the timer so that it does not activate until N steps or seconds
            # have passed.
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            self._file_writer.add_run_metadata(run_values.run_metadata, "step_%d" % global_step)

        self._next_step = global_step + 1


def matthews_correlation(y_true, y_pred):
    cm = tf.confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2 and cm.shape[1] == 2:
        tp = cm[0][0]
        tn = cm[1][1]

        fp = cm[0][1]
        fn = cm[1][0]
    else:
        tp = tf.constant(1E-9, dtype=tf.float32)
        tn = tf.constant(1E-9, dtype=tf.float32)
        fp = tf.constant(1E-9, dtype=tf.float32)
        fn = tf.constant(1E-9, dtype=tf.float32)

    numerator = tf.cast((tp * tn - fp * fn), tf.float32)
    denominator = tf.sqrt(tf.cast((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), tf.float32))

    result = tf.divide(numerator, denominator)

    return result, tf.group(tp, tn, fp, fn)


def custom_metrics(labels, predictions):
    return {
        'matthews_correlation': matthews_correlation(labels, predictions['class_ids'])
        # 'mean_labels': tf.metrics.mean(labels),
        # 'mean_predictions': tf.metrics.mean(predictions['class_ids'])
    }


def lstm_with_attention_model_fn(features, labels, mode, params):
    input_layer = features['stats']
    # reshape to [batches, time series, channels]
    input_layer = tf.reshape(input_layer, shape=[-1, 3040, 1])
    # transpose to time-major form [time series, batches, channels]
    input_layer = tf.transpose(input_layer, [1, 0, 2])
    lstm1, _ = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=128, direction='unidirectional')(input_layer)
    # attention_mechanism = tf.contrib.seq2seq.LuongMonotonicAttention(num_units=input_layer.shape[2],
    #                                                                  memory=lstm1)
    # lstm2 = tf.contrib.seq2seq.AttentionWrapper(
    #     cell=tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=64, direction='unidirectional'),
    #     attention_mechanism=attention_mechanism
    # )(lstm1)
    lstm2, _ = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=64, direction='unidirectional')(lstm1)

    lstm2 = tf.reduce_sum(lstm2, axis=2)

    # transpose to channels major form
    lstm2 = tf.transpose(lstm2, [1, 0])

    # mask = tf.tile(
    #     tf.expand_dims(tf.sequence_mask(3040, tf.shape(lstm2)[1]), 2),
    #     [1, 1, tf.shape(lstm2)[2]])
    # zero_outside = tf.where(mask, lstm2, tf.zeros_like(lstm2))
    # lstm2 = tf.reduce_sum(zero_outside, axis=1)

    dense = tf.layers.dense(inputs=lstm2, units=64, activation=tf.nn.leaky_relu)
    logits = tf.layers.dense(inputs=dense, units=params['num_classes'], activation=tf.nn.softmax)

    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    one_hot_labels = tf.one_hot(labels, 2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
    tf.summary.scalar('OptimizeLoss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels, predicted_indices),
                           'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(labels,
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
    # target_indices = tf.argmax(input=labels, axis=1)
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
