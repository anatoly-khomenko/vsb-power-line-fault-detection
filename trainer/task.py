import argparse
import os

import tensorflow as tf
from tqdm import tqdm
from trainer import create_tf_records

from trainer import model
from trainer import dataset

import multiprocessing


def run_prediction(estimator, predict_input_fn, start_id, output_file_path):
    predictions_iterator = estimator.predict(input_fn=predict_input_fn)
    signal_id = start_id
    total_positives_predicted = 0

    with open(output_file_path, 'w') as submission_file:
        submission_file.write('signal_id,target\n')
        for prediction_dict in tqdm(predictions_iterator):
            class_id = prediction_dict['class_ids']
            total_positives_predicted = total_positives_predicted + class_id
            submission_file.write('%d,%d\n' % (signal_id, class_id))
            signal_id = signal_id + 1
    tf.logging.log(tf.logging.INFO, 'Total positives predicted: %d' % total_positives_predicted)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # repeatable results for fake data
    if args.use_fake_data:
        tf.set_random_seed(12345)

    cpu_count = multiprocessing.cpu_count()

    train_files = tf.gfile.Glob(os.path.join(args.data_dir, "train-*.tfrecords"))
    if len(train_files) == 0:
        tf.logging.log(tf.logging.INFO, 'Converting train data to TFRecord format:')
        create_tf_records.convert_to_tf(signals_file_name=os.path.join(args.arrow_data_dir, 'train.parquet'),
                                        metadata_file_name=os.path.join(args.arrow_data_dir, 'metadata_train.csv'),
                                        target_tf_prefix=os.path.join(args.data_dir, 'train'),
                                        batch_size=args.train_batch_size)
        train_files = tf.gfile.Glob(os.path.join(args.data_dir, 'train-*.tfrecords'))

    test_files = tf.gfile.Glob(os.path.join(args.data_dir, 'test-*.tfrecords'))
    if len(test_files) == 0:
        tf.logging.log(tf.logging.INFO, 'Converting test data to TFRecord format:')
        create_tf_records.convert_to_tf(signals_file_name=os.path.join(args.arrow_data_dir, 'test.parquet'),
                                        metadata_file_name=os.path.join(args.arrow_data_dir, 'metadata_test.csv'),
                                        target_tf_prefix=os.path.join(args.data_dir, 'test'),
                                        batch_size=args.train_batch_size)
        test_files = tf.gfile.Glob(os.path.join(args.data_dir, 'test-*.tfrecords'))

    # Augment data: add positive samples and save result for future runs.
    positive_train_records = 525
    total_train_records = 8712
    stats_folder = os.path.join(args.data_dir, 'stats')
    os.makedirs(stats_folder, exist_ok=True)
    preprocessed_train_files = tf.gfile.Glob(os.path.join(stats_folder, 'train-*.tfrecords'))
    if len(preprocessed_train_files) == 0:
        tf.logging.log(tf.logging.INFO, 'Preprocessing train data.')
        augmentation_need = (total_train_records - positive_train_records) // positive_train_records
        train_all = dataset.load(train_files)
        positives = train_all.filter(dataset.positives)
        flipped = positives.map(dataset.flip_map_func, num_parallel_calls=cpu_count)
        augmented = positives.map(dataset.roll_map_func, num_parallel_calls=cpu_count)
        for i in range(0, augmentation_need - 1, 2):
            augmented = augmented.concatenate(positives.map(dataset.roll_map_func, num_parallel_calls=cpu_count))
            augmented = augmented.concatenate(flipped.map(dataset.roll_map_func, num_parallel_calls=cpu_count))
        train_all = train_all.concatenate(augmented)

        train_all = train_all.batch(args.preprocess_batch_size)\
            .map(map_func=dataset.stats_map_func, num_parallel_calls=cpu_count).shuffle(buffer_size=20000)
        dataset.generate_stats(stats_folder, train_all, 'train')
        tf.logging.log(tf.logging.INFO, 'Train data preprocessing completed.')
        preprocessed_train_files = tf.gfile.Glob(os.path.join(stats_folder, 'train-*.tfrecords'))

    preprocessed_test_files = tf.gfile.Glob(os.path.join(stats_folder, 'test-*.tfrecords'))
    if len(preprocessed_test_files) == 0:
        tf.logging.log(tf.logging.INFO, 'Preprocessing test data.')
        test_all = dataset.load(test_files).batch(args.preprocess_batch_size)\
            .map(map_func=dataset.stats_map_func, num_parallel_calls=cpu_count)
        dataset.generate_stats(stats_folder, test_all, 'test')
        tf.logging.log(tf.logging.INFO, 'Test data preprocessing completed.')
        preprocessed_test_files = tf.gfile.Glob(os.path.join(stats_folder, 'test-*.tfrecords'))

    # total_train_records = 2560
    eval_split = int(args.train_eval_split * total_train_records / 100)
    total_eval_records = total_train_records - eval_split

    tf.logging.log(tf.logging.INFO, 'Train records: %d, Eval records: %d' % (eval_split, total_eval_records))

    # profiler_hook = tf.train.ProfilerHook(save_steps=args.save_summary_steps,
    #                                      output_dir=os.path.join(args.output_dir, 'profiler'),
    #                                      show_dataflow=True,
    #                                      show_memory=True)

    if args.save_session_metadata:
        train_hooks = [
            model.MetadataProfilerHook(save_steps=args.save_summary_steps,
                                       output_dir=args.output_dir)
        ]
    else:
        train_hooks = None

    def train_input_fn():
        return dataset.load(preprocessed_train_files)\
            .take(eval_split)\
            .batch(args.train_batch_size).repeat()\
            .prefetch(buffer_size=None)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.train_steps, hooks=train_hooks)

    features = {
      'stats_1': tf.placeholder(dtype=tf.float32, shape=3040)
    }
    final_exporter = tf.estimator.FinalExporter('finalExporter',
                                                tf.estimator.export.build_raw_serving_input_receiver_fn(features))
    os.makedirs(os.path.join(args.output_dir, 'export/finalExporter'), exist_ok=True)

    def eval_input_fn():
        return dataset.load(preprocessed_train_files)\
            .skip(eval_split)\
            .batch(args.eval_batch_size)\
            .prefetch(buffer_size=None)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=int(total_eval_records / args.eval_batch_size) + 1,
                                      exporters=final_exporter,
                                      start_delay_secs=args.eval_delay_secs,
                                      throttle_secs=10)

    # h_params = {
    #     'feature_columns': feature_columns,
    #     'filters_1': 2, 'kernel_size_1': 9,
    #     'filters_2': 2, 'kernel_size_2': 9,
    #     'pool_size_1': 11, 'strides_1': 1,
    #     'filters_3': 2, 'kernel_size_3': 9,
    #     'filters_4': 2, 'kernel_size_4': 9,
    #     'pool_size_2': 11, 'strides_2': 1,
    #     'dropout_rate': 0.5,
    #     'num_classes': 2,
    #     'learning_rate': 0.001
    # }

    # session_config = tf.ConfigProto(device_count={'GPU': 0})
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
    #    model_fn=model.lstm_with_attention_model_fn,
    #    model_dir=args.output_dir,
    #    config=run_config,
    #    params=h_params)

    # k_model = keras_model.model_lstm([eval_split, 160, 19])
    # estimator = keras_model.get_estimator(model=k_model, model_dir=args.output_dir, run_config=run_config)

    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # estimator = tf.estimator.LinearClassifier(feature_columns,
    #                                           model_dir=args.output_dir,
    #                                           config=run_config)
    feature_columns = [
        tf.feature_column.numeric_column(key='stats_1', shape=3040, dtype=tf.float32),
    ]
    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[2048, 1024, 512],
        model_dir=args.output_dir,
        config=run_config,
        batch_norm=False,
        dropout=0.7)

    estimator = tf.contrib.estimator.add_metrics(estimator, model.custom_metrics)
    evaluation_metrics, export_results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    tf.logging.log(tf.logging.INFO, evaluation_metrics)
    tf.logging.log(tf.logging.INFO, export_results)

    def test_input_fn():
        return dataset.load(preprocessed_test_files)\
            .batch(args.eval_batch_size)\
            .prefetch(buffer_size=None)

    run_prediction(estimator=estimator, predict_input_fn=test_input_fn, start_id=total_train_records,
                   output_file_path=os.path.join(args.output_dir, 'submission.csv'))

    def all_train_input_fn():
        return dataset.load(preprocessed_train_files)\
            .batch(args.eval_batch_size)\
            .prefetch(buffer_size=None)

    run_prediction(estimator=estimator, predict_input_fn=all_train_input_fn, start_id=0,
                   output_file_path=os.path.join(args.output_dir, 'train-prediction.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--arrow_data_dir',
        help='GCS or local path to training data in pyarrow format',
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
        '--preprocess_batch_size',
        help='Batch size for data preprocessing',
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
