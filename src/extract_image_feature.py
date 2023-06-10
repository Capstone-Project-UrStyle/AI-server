"""Run the inference of Bi-LSTM model given input images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import pickle as pkl
import numpy as np


FLAGS = tf.flags.FLAGS


def run(all_items, inference_model, inference_saver, inference_session):
  # Restore session to checkpoint
  inference_saver.restore(inference_session, FLAGS.checkpoint_path)
  
  # Save image ids and features in a dictionary.
  image_features = dict()
  count = 0
  for item in all_items:
    image_path = item['image']
    if (count % 1000 == 0): print('Processed %d images...' % (count))
    with tf.gfile.GFile(image_path, "r") as f:
        image_feed = f.read()

    [feat, rnn_feat] = inference_session.run([inference_model.image_embeddings,
                                              inference_model.rnn_image_embeddings],
                                              feed_dict={"image_feed:0": image_feed})
    
    image_features[image_path] = dict()
    image_features[image_path]["image_rnn_feat"] = np.squeeze(rnn_feat)
    image_features[image_path]["image_feat"] = np.squeeze(feat)
    
    count += 1

  # Extract feature_file
  with open(FLAGS.feature_file, "wb") as f:
    pkl.dump(image_features, f)
  
  print("Extract %d images's features successfully!" % (count))

  return 'Extract all item image features successfully!'


def extract_new_item_image_features(new_item_image_path, image_features, inference_model, inference_saver, inference_session):
  # Check if image file is exist
  if os.path.isfile(new_item_image_path):
    # Restore session to checkpoint
    inference_saver.restore(inference_session, FLAGS.checkpoint_path)

    # Extract new item image features
    with tf.gfile.GFile(new_item_image_path, "r") as f:
      image_feed = f.read()

    [feat, rnn_feat] = inference_session.run([inference_model.image_embeddings,
                                              inference_model.rnn_image_embeddings],
                                              feed_dict={"image_feed:0": image_feed})
    
    image_features[new_item_image_path] = dict()
    image_features[new_item_image_path]["image_rnn_feat"] = np.squeeze(rnn_feat)
    image_features[new_item_image_path]["image_feat"] = np.squeeze(feat)

    # Delete old image_features file
    os.remove(FLAGS.feature_file)

    # Replace with new image_features file
    with open(FLAGS.feature_file, "wb") as f:
      pkl.dump(image_features, f)

    return "Extract new item image features successfully!"
  else:
    return "Item image file not exists!"


def remove_item_image_features(item_image_path, image_features):
  # Check if image file is exist
  if os.path.isfile(item_image_path):
    # Remove item image path key in feature file
    if item_image_path in image_features:
      del image_features[item_image_path]
    else:
      return 'Item image path not exists in image features file!'

    # Delete old image_features file
    os.remove(FLAGS.feature_file)

    # Replace with new image_features file
    with open(FLAGS.feature_file, "wb") as f:
      pkl.dump(image_features, f)

    return "Remove item image features successfully!"
  else:
    return "Item image file not exists!"