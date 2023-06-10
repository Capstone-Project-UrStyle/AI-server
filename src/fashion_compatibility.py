"""Predict the fashion compatibility of a given image sequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn import metrics


tf.flags.DEFINE_integer("direction", 2,
                        "2: bidirectional; 1: forward only; -1: backward only.")

FLAGS = tf.flags.FLAGS


def run_compatibility_inference(inference_session, image_seqs, image_feats, num_lstm_units, inference_model):
  emb_seqs = image_feats[image_seqs,:]
  num_images = float(len(image_seqs))
  zero_state = np.zeros([1, 2 * num_lstm_units])
  f_score = 0
  b_score = 0

  if FLAGS.direction != -1:
    # Forward RNN.
    outputs = []
    input_feed = np.reshape(emb_seqs[0], [1,-1])
    # Run first step with all zeros initial state.
    [lstm_state, lstm_output] = inference_session.run(fetches=["lstm/f_state:0","f_logits/f_logits/BiasAdd:0"],
                                                      feed_dict={"lstm/f_input_feed:0":input_feed,
                                                                 "lstm/f_state_feed:0":zero_state})
    outputs.append(lstm_output)

    # Run remaining steps.
    for step in range(int(num_images)-1):
      input_feed = np.reshape(emb_seqs[step+1], [1,-1])
      [lstm_state, lstm_output] = inference_session.run(fetches=["lstm/f_state:0","f_logits/f_logits/BiasAdd:0"],
                                                        feed_dict={"lstm/f_input_feed:0":input_feed,
                                                                   "lstm/f_state_feed:0":lstm_state})
      outputs.append(lstm_output)
    
    # Calculate the loss.
    # Different from the training process where the loss is calculated in each
    # mini batch, during testing, we get the loss againist the whole test set.
    # This is pretty slow, maybe a better method could be used.
    s = np.squeeze(np.dot(np.asarray(outputs), np.transpose(image_feats)))
    f_score = inference_session.run(inference_model.lstm_xent_loss,
                                    feed_dict={"lstm/pred_feed:0":s,
                                               "lstm/next_index_feed:0":image_seqs[1:] + [image_feats.shape[0]-1]})
    
    f_score = - np.mean(f_score)
    
  if FLAGS.direction != 1:
    # Backward RNN.
    outputs = []
    input_feed = np.reshape(emb_seqs[-1], [1,-1])
    [lstm_state, lstm_output] = inference_session.run(fetches=["lstm/b_state:0","b_logits/b_logits/BiasAdd:0"],
                                                      feed_dict={"lstm/b_input_feed:0":input_feed,
                                                                 "lstm/b_state_feed:0":zero_state})
    outputs.append(lstm_output)
    
    # Run remaining steps.
    for step in range(int(num_images)-1):
      input_feed = np.reshape(emb_seqs[int(num_images)-2-step], [1,-1])
      [lstm_state, lstm_output] = inference_session.run(fetches=["lstm/b_state:0","b_logits/b_logits/BiasAdd:0"],
                                                        feed_dict={"lstm/b_input_feed:0":input_feed,
                                                                   "lstm/b_state_feed:0":lstm_state})
      outputs.append(lstm_output)
    
    # Calculate the loss.
    s = np.squeeze(np.dot(np.asarray(outputs), np.transpose(image_feats)))
    b_score = inference_session.run(inference_model.lstm_xent_loss,
                                    feed_dict={"lstm/pred_feed:0":s,
                                               "lstm/next_index_feed:0": image_seqs[-2::-1] + [image_feats.shape[0]-1]})
    b_score = - np.mean(b_score)
  
  return [f_score, b_score]

  
def run(item_sets, image_features, inference_model_config, inference_model, inference_saver, inference_session):
  # Restore session to checkpoint
  inference_saver.restore(inference_session, FLAGS.checkpoint_path)

  image_paths = image_features.keys()
  
  image_feats = np.zeros((len(image_paths) + 1,
                  len(image_features[image_paths[0]]["image_rnn_feat"])))
  
  # image_feats has one more zero vector as the representation of END of RNN prediction.
  for i, image_path in enumerate(image_paths):
    # Image feature in the RNN space.
    image_feats[i] = image_features[image_path]["image_rnn_feat"]
  
  all_f_scores = []
  all_b_scores = []
  all_scores = []

  for item_set in item_sets:
    image_seqs = []
    for item in item_set:
      image_seqs.append(image_paths.index(item['image']))
      
    [f_score, b_score] = run_compatibility_inference(inference_session, image_seqs, image_feats,
                                                     inference_model_config.num_lstm_units, inference_model)
    
    all_f_scores.append(f_score)
    all_b_scores.append(b_score)
    all_scores.append(f_score + b_score)
    
  # Calculate AUC and AP      
  # fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
  # print("Compatibility AUC: %f for %d outfits" %
  #         (metrics.auc(fpr, tpr), len(all_labels)))

  # with open(FLAGS.result_file, "wb") as f:
  #   pkl.dump({"all_f_scores": all_f_scores, "all_b_scores": all_b_scores}, f)
  
  return all_scores
