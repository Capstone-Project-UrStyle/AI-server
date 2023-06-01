#!/bin/bash
CHECKPOINT_DIR="models/model.ckpt-34865"

python -B src/run_inference.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --item_ids="data/label/test_no_dup.json" \
  --image_dir="../../Back-end/public/images/items/" \
  --feature_file="data/features/image_features.pkl" \
  --rnn_type="lstm"