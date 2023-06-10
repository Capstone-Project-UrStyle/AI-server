
"""Given multimodal queries, complete the outfit wiht bi-LSTM and VSE model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

import pickle as pkl
import tensorflow as tf
import numpy as np


tf.flags.DEFINE_float("balance_factor", 5.0,
                      "Trade off between image and text input."
                      "Larger balance_factor encourages higher correlation with text query")

FLAGS = tf.flags.FLAGS

def norm_row(a):
  """L2 normalize each row of a given set."""
  try:
    return a / np.linalg.norm(a, axis=1)[:, np.newaxis]
  except:
    return a / np.linalg.norm(a)

def rnn_one_step(inference_session, input_feed, lstm_state, direction='f'):
  """Run one step of the RNN."""
  if direction == 'f':
    # Forward
    [lstm_state, lstm_output] = inference_session.run(
        fetches=['lstm/f_state:0', 'f_logits/f_logits/BiasAdd:0'],
        feed_dict={'lstm/f_input_feed:0': input_feed,
                   'lstm/f_state_feed:0': lstm_state})
  else:
    # Backward
    [lstm_state, lstm_output] = inference_session.run(
        fetches=['lstm/b_state:0', 'b_logits/b_logits/BiasAdd:0'],
        feed_dict={'lstm/b_input_feed:0': input_feed,
                   'lstm/b_state_feed:0': lstm_state})
    
  return lstm_state, lstm_output


def run_forward_rnn(inference_session, image_idx, image_feats, num_lstm_units):
  """ Run forward RNN given a query."""
  res_set = []
  lstm_state = np.zeros([1, 2 * num_lstm_units])
  for image_id in image_idx:
    input_feed = np.reshape(image_feats[image_id], [1, -1])
    # Run first step with all zeros initial state.
    [lstm_state, lstm_output] = rnn_one_step(inference_session, input_feed, lstm_state, direction='f')

  # Maximum length of the outfit is set to 10.
  for step in range(10):
    curr_score = np.exp(np.dot(lstm_output, np.transpose(image_feats)))
    curr_score /= np.sum(curr_score)

    next_image = np.argsort(-curr_score)[0][0]
    # 0.00001 is used as a probablity threshold to stop the generation.
    # i.e, if the prob of end-of-set is larger than 0.00001, then stop.
    if next_image == image_feats.shape[0] - 1 or curr_score[0][-1] > 0.00001:
      # print('OVER')
      break
    else:
      input_feed = np.reshape(image_feats[next_image], [1, -1])
      [lstm_state, lstm_output] = rnn_one_step(inference_session, input_feed, lstm_state, direction='f')
      res_set.append(next_image)

  return res_set


def run_backward_rnn(sess, image_idx, image_feats, num_lstm_units):
  """ Run backward RNN given a query."""
  res_set = []
  lstm_state = np.zeros([1, 2 * num_lstm_units])
  for image_id in reversed(image_idx):
    input_feed = np.reshape(image_feats[image_id], [1, -1])
    [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='b')
  for step in range(10):
    curr_score = np.exp(np.dot(lstm_output, np.transpose(image_feats)))
    curr_score /= np.sum(curr_score)
    next_image = np.argsort(-curr_score)[0][0]
    # 0.00001 is used as a probablity threshold to stop the generation.
    # i.e, if the prob of end-of-set is larger than 0.00001, then stop.
    if next_image == image_feats.shape[0] - 1 or curr_score[0][-1] > 0.00001:
      # print('OVER')
      break
    else:
      input_feed = np.reshape(image_feats[next_image], [1, -1])
      [lstm_state, lstm_output] = rnn_one_step(
          sess, input_feed, lstm_state, direction='b')
      res_set.append(next_image)

  return res_set


def run_fill_rnn(inference_session, start_id, end_id, num_blank, image_feats, num_lstm_units):
  """Fill in the blanks between start and end."""
  if num_blank == 0:
    return [start_id, end_id]
  lstm_state = np.zeros([1, 2 * num_lstm_units])
  input_feed = np.reshape(image_feats[start_id], [1, -1])
  [lstm_state, lstm_output] = rnn_one_step(inference_session, input_feed, lstm_state, direction='f')

  f_outputs = []
  for i in range(num_blank):
    f_outputs.append(lstm_output[0])
    curr_score = np.exp(np.dot(lstm_output, np.transpose(image_feats)))
    curr_score /= np.sum(curr_score)
    next_image = np.argsort(-curr_score)[0][0]
    input_feed = np.reshape(image_feats[next_image], [1, -1])
    [lstm_state, lstm_output] = rnn_one_step(inference_session, input_feed, lstm_state, direction='f')

  lstm_state = np.zeros([1, 2 * num_lstm_units])
  input_feed = np.reshape(image_feats[end_id], [1, -1])
  [lstm_state, lstm_output] = rnn_one_step(inference_session, input_feed, lstm_state, direction='b')

  b_outputs = []
  for i in range(num_blank):
    b_outputs.insert(0, lstm_output[0])
    curr_score = np.exp(np.dot(lstm_output, np.transpose(image_feats)))
    curr_score /= np.sum(curr_score)
    next_image = np.argsort(-curr_score)[0][0]
    input_feed = np.reshape(image_feats[next_image], [1, -1])
    [lstm_state, lstm_output] = rnn_one_step(inference_session, input_feed, lstm_state, direction='b')

  outputs = np.asarray(f_outputs) + np.asarray(b_outputs)
  score = np.exp(np.dot(outputs, np.transpose(image_feats)))

  score /= np.sum(score, axis=1)[:, np.newaxis]
  blank_ids = np.argmax(score, axis=1)
  return [start_id] + list(blank_ids) + [end_id]


def run_set_inference(inference_session, query_item_image_paths, image_paths, image_feats, num_lstm_units):
  image_idx = []
  for query_item_image_path in query_item_image_paths:
    try:
      image_idx.append(image_paths.index(query_item_image_path))
    except:
      print('not found')
      return

  # Dynamic search
  # Run the whole bi-LSTM on the first item
  first_f_set = run_forward_rnn(inference_session, image_idx[:1], image_feats, num_lstm_units)
  first_b_set = run_backward_rnn(inference_session, image_idx[:1], image_feats, num_lstm_units)

  first_posi = len(first_b_set)
  first_set = first_b_set + image_idx[:1] + first_f_set

  image_set = []
  for i in first_set:
    image_set.append(image_paths[i])

  if len(query_item_image_paths) >= 2:
    current_set = norm_row(image_feats[first_set, :])
    all_position = [first_posi]
    for image_id in image_idx[1:]:
      # Gradually adding items into it
      # Find nn of the next item
      insert_posi = np.argmax(np.dot(norm_row(image_feats[image_id, :]), np.transpose(current_set)))
      all_position.append(insert_posi)

    # Run bi LSTM to fill items between first item and this item
    start_posi = np.min(all_position)
    end_posi = np.max(all_position)

    sets = run_fill_rnn(inference_session, image_idx[0], image_idx[1],
                        end_posi - start_posi - 1, image_feats, num_lstm_units)

  else:
    # Run bi LSTM again
    sets = image_idx
  f_set = run_forward_rnn(inference_session, sets, image_feats, num_lstm_units)
  b_set = run_backward_rnn(inference_session, sets, image_feats, num_lstm_units)

  image_set = []
  for i in b_set[::-1] + sets + f_set:
    image_set.append(image_paths[i])

  return b_set[::-1] + sets + f_set


def nn_search(i, image_embs, word_vec):
  # score = np.dot(test_emb, np.transpose(test_emb[i] + word_vec))
  score = np.dot(image_embs, np.transpose(image_embs[i] + FLAGS.balance_factor * word_vec))
  return np.argmax(score)


def run(query_item_image_paths, query_keywords, words,
        image_features, inference_model_config, inference_model, inference_saver, inference_session):
  # Restore session to checkpoint
  inference_saver.restore(inference_session, FLAGS.checkpoint_path)

  image_paths = image_features.keys() # image_paths
  image_feats = np.zeros((len(image_paths) + 1,
                          len(image_features[image_paths[0]]["image_rnn_feat"])))
  image_embs = np.zeros((len(image_paths),
                         len(image_features[image_paths[0]]["image_feat"])))

  for i, image_path in enumerate(image_paths):
    # Image feature in the RNN space.
    image_feats[i] = image_features[image_path]["image_rnn_feat"]
    # Image feature in the joint embedding space.
    image_embs[i] = image_features[image_path]["image_feat"]

  image_embs = norm_row(image_embs)

  # Get the word embedding.
  [word_emb] = inference_session.run([inference_model.embedding_map])

  # Calculate the embedding of the query keyword
  # Run Bi-LSTM model using the query item images
  rnn_sets = run_set_inference(inference_session, query_item_image_paths, image_paths,
                                image_feats, inference_model_config.num_lstm_units)
  
  print("RNN recommend outfit before applying keyword:", rnn_sets)

  recommendation_results = []

  # Reranking the LSTM prediction with similarity with the query keyword        
  for query_keyword in query_keywords:
    if query_keyword != "":
      # Get the indices of query images.
      image_idx = []
      for query_item_image_path in query_item_image_paths:
        try:
          image_idx.append(image_paths.index(query_item_image_path))
        except:
          print('not found')
          return

      outfit_recommendations = np.array(rnn_sets)
      
      # Calculate the word embedding
      query_keyword = [i+1 for i in range(len(words)) if words[i] in query_keyword.split()]
      query_emb = norm_row(np.sum(word_emb[query_keyword], axis=0))
      for i, j in enumerate(outfit_recommendations):
        if j not in image_idx:
          outfit_recommendations[i] = nn_search(j, image_embs, query_emb)
      
      print("RNN recommend outfit after applying keyword:", outfit_recommendations)
      outfit_recommendation_image_paths = list(set([image_paths[i] for i in outfit_recommendations]))

      recommendation_results.append([item_image_path for item_image_path in outfit_recommendation_image_paths 
                                      if item_image_path not in query_item_image_paths])

  return recommendation_results
