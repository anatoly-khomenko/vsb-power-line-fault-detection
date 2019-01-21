import argparse
import os

import tensorflow as tf
from trainer import model


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

    tp = cm[0][0]
    tn = cm[1][1]

    fp = cm[0][1]
    fn = cm[1][0]

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


def run_prediction(estimator, predict_input_fn, start_id, output_file_path):
    predictions_iterator = estimator.predict(input_fn=predict_input_fn)
    signal_id = start_id
    total_positives_predicted = 0

    with open(output_file_path, 'w') as submission_file:
        submission_file.write('signal_id,target\n')
        for prediction_dict in predictions_iterator:
            class_id = prediction_dict['class_ids'][0]
            total_positives_predicted = total_positives_predicted + class_id
            submission_file.write('%d,%d\n' % (signal_id, class_id))
            signal_id = signal_id + 1
    print("Total positives predicted: ", total_positives_predicted)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # repeatable results for fake data
    if args.use_fake_data:
        tf.set_random_seed(12345)

    all_train_files = tf.gfile.Glob(os.path.join(args.data_dir, "Inception-V3-features-train.tfrecords"))
    # number_of_train_files = int(args.train_eval_split*len(all_train_files)/100)
    # train_files = all_train_files[:number_of_train_files]
    # eval_files = all_train_files[number_of_train_files+1:]
    predict_files = tf.gfile.Glob(os.path.join(args.data_dir, "Inception-V3-features-test.tfrecords"))
    # print("Eval files:", eval_files)

    total_train_records = 8712
    eval_split = int(args.train_eval_split * total_train_records / 100)
    total_eval_records = total_train_records - eval_split

    print("Train records: %d, Eval records: %d" % (eval_split, total_eval_records))

    train_input_fn = model.get_input_fn(filename_queue=all_train_files, batch_size=args.train_batch_size,
                                        take_count=eval_split, pos_weight=10.0, fake=args.use_fake_data)
    eval_input_fn = model.get_input_fn(filename_queue=all_train_files, batch_size=args.eval_batch_size,
                                       skip_count=eval_split, pos_weight=1.0, fake=args.use_fake_data)
    predict_input_fn = model.get_input_fn(filename_queue=predict_files, batch_size=args.eval_batch_size,
                                          predict=True, fake=args.use_fake_data)

    # profiler_hook = tf.train.ProfilerHook(save_steps=args.save_summary_steps,
    #                                      output_dir=os.path.join(args.output_dir, 'profiler'),
    #                                      show_dataflow=True,
    #                                      show_memory=True)

    if args.save_session_metadata:
        train_hooks = [
            MetadataProfilerHook(save_steps=args.save_summary_steps,
                                 output_dir=args.output_dir)
        ]
    else:
        train_hooks = None

    os.makedirs(os.path.join(args.output_dir, 'export/finalExporter'), exist_ok=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.train_steps, hooks=train_hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=int(total_eval_records/args.eval_batch_size) + 1,
                                      exporters=tf.estimator.FinalExporter('finalExporter', model.serving_input_fn),
                                      start_delay_secs=args.eval_delay_secs,
                                      throttle_secs=10)

    feature_columns = [
        tf.feature_column.numeric_column(key='signal', shape=2048),
        tf.feature_column.numeric_column(key='weight', default_value=1)
    ]
    h_params = {
        'feature_columns': feature_columns,
        'filters_1': 2, 'kernel_size_1': 9,
        'filters_2': 2, 'kernel_size_2': 9,
        'pool_size_1': 11, 'strides_1': 1,
        'filters_3': 2, 'kernel_size_3': 9,
        'filters_4': 2, 'kernel_size_4': 9,
        'pool_size_2': 11, 'strides_2': 1,
        'dropout_rate': 0.5,
        'num_classes': 2,
        'learning_rate': 0.001
    }

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
                                                              allocator_type="BFC"),
                                    allow_soft_placement=True,
                                    log_device_placement=False)

    # Workaround for NVidia NCCL library absence in TF 1.12
    # https://stackoverflow.com/questions/54077118/google-cloud-ml-engine-fails-at-loading-libnccl
    if '.12.' in tf.__version__:
        distribution = None
        tf.logging.info("TF 1.12 detected, not able to use multi-GPU training.")
    else:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=4, prefetch_on_device=True)
        tf.logging.info("Using MirroredStrategy for multi-GPU training.")

    run_config = tf.estimator.RunConfig(save_summary_steps=args.save_summary_steps,
                                        save_checkpoints_steps=args.save_checkpoints_steps,
                                        log_step_count_steps=args.log_step_count_steps,
                                        session_config=session_config,
                                        train_distribute=distribution)

    # estimator = tf.estimator.Estimator(
    #    model_fn=model.dnn_unbalanced_classifier_model_fn,
    #    model_dir=args.output_dir,
    #    config=run_config,
    #    params=h_params)

    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # estimator = tf.estimator.LinearClassifier(feature_columns,
    #                                           model_dir=args.output_dir,
    #                                           config=run_config)

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        weight_column='weight',
        hidden_units=[1024, 512],
        model_dir=args.output_dir,
        config=run_config,
        batch_norm=True,
        dropout=0.3)

    estimator = tf.contrib.estimator.add_metrics(estimator, custom_metrics)

    # n_batches_per_layer = 0.5 * eval_split / args.train_batch_size
    # estimator = tf.estimator.BoostedTreesClassifier(
    #     feature_columns=feature_columns,
    #     n_batches_per_layer=n_batches_per_layer,
    #     model_dir=args.output_dir,
    #     config=run_config)

    evaluation_metrics, export_results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print(evaluation_metrics)
    print(export_results)

    # estimator.train(input_fn=train_input_fn, hooks=train_hooks, max_steps=args.train_steps)
    # evaluation_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=args.eval_steps)

    run_prediction(estimator, predict_input_fn, total_train_records, os.path.join(args.output_dir, 'submission.csv'))
    all_train_input_fn = model.get_input_fn(filename_queue=all_train_files, batch_size=args.train_batch_size,
                                            fake=args.use_fake_data, predict=True)
    run_prediction(estimator, all_train_input_fn, 0, os.path.join(args.output_dir, 'train-prediction.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--train_eval_split',
        help='Percentage of train-*.tfrecords files to use for training. Remainder goes to evaluation set.',
        type=int,
        default=75
    )
    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=5
    )
    parser.add_argument(
        '--eval_batch_size',
        help='Batch size for evaluation steps',
        type=int,
        default=5
    )
    parser.add_argument(
        '--train_steps',
        help='Steps to run the training job for.',
        type=int,
        default=100
    )
    parser.add_argument(
        '--save_summary_steps',
        help='Steps between saving the summary.',
        type=int,
        default=10
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        help='Steps between saving checkpoint.',
        type=int,
        default=25
    )
    parser.add_argument(
        '--log_step_count_steps',
        help='Steps between logging step count.',
        type=int,
        default=10
    )
    parser.add_argument(
        '--eval_steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=10,
        type=int
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--save_session_metadata',
        action="store_true",
        help='Log memory usage for each checkpoint',
    )
    parser.add_argument(
        '--use_fake_data',
        action="store_true",
        help='create fake data which are easy to learn with the same amount of positive and negative samples as in '
             'real data',
    )

    main(parser.parse_args())
