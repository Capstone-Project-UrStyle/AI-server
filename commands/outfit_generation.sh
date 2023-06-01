#!/bin/bash
CHECKPOINT_DIR="models/model.ckpt-34865"

# Run inference on images.
python -B src/set_generation.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --image_dir="../../Back-end/public/images/items/" \
  --feature_file="data/features/image_features.pkl" \
  --query_file="commands/outfit_generation_query.json" \
  --word_dict_file="data/final_word_dict.txt" \
  --result_dir="outfit_generate_results"

chmod 777 outfit_generate_results/*
  