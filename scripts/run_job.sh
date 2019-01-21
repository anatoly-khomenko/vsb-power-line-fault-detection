#!/usr/bin/env bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT_ID}-mlengine"

JOB_ID="${USER}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="anatoly_khomenko_20190110"

gcloud ml-engine jobs submit training ${JOB_ID} \
      --package-path trainer \
      --module-name trainer.task \
      --staging-bucket ${BUCKET} \
      --job-dir ${BUCKET}/${OUTPUT_DIR} \
      --python-version 3.5 \
      --runtime-version 1.11 \
      --region us-central1 \
      --config config/config.yaml \
      -- \
      --data_dir ${BUCKET}/tf \
      --output_dir ${BUCKET}/${OUTPUT_DIR} \
      --eval_batch_size 10 \
      --train_batch_size 10 \
      --train_steps 11000 \
      --save_checkpoints_steps 2000 \
      --eval_steps 400 \
      --save_summary_steps 100