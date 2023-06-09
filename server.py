from flask import Flask, request
from flask_cors import CORS

import os
import json
import tensorflow as tf
import itertools

from src import configuration
from src import polyvore_model_bi as polyvore_model

from src import extract_image_feature
from src import outfit_generation
from src import fashion_compatibility

app = Flask(__name__)
CORS(app, resource={r"/*": {"origins": "*"}})

app.secret_key = "secrect_key"

words = None

inference_model_config = None
inference_model = None
inference_saver = None
inference_session = None


tf.flags.DEFINE_string("checkpoint_path", "models/model.ckpt-34865",
                       "Model checkpoint file or directory containing a model checkpoint file.")
tf.flags.DEFINE_string("image_dir", "public/images/items/",
                      "Directory containing images.")
tf.flags.DEFINE_string("feature_file", "data/features/image_features.pkl",
                      "Directory to save the features")
tf.flags.DEFINE_string("rnn_type", "lstm", "Type of RNN.")


def extract_all_item_image_features():
    all_item_image_paths = []
    folder_path = 'public/images/items/'
    file_names = os.listdir(folder_path)

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            all_item_image_paths.append(file_path)
    
    all_items = [{'image': image_path} for image_path in all_item_image_paths]

    if (inference_model_config != None and
        inference_model != None and
        inference_saver != None and
        inference_session != None):
        
        # Extract all items's image feature
        extract_feature_result = extract_image_feature.run(all_items,
                                                           inference_model,
                                                           inference_saver,
                                                           inference_session)
        
        return extract_feature_result
    else:
        return 'Something went wrong!'


@app.route('/extract-new-item-image-features', methods=['POST'])
def extract_new_item_image_features():
    try:
        # Get new item data from request
        new_item_image_path = request.get_json().get('new_item_image_path')

        if (new_item_image_path != None and
            inference_model_config != None and
            inference_model != None and
            inference_saver != None and
            inference_session != None):
            
            # Extract all items's image feature
            extract_feature_result = extract_image_feature.extract_new_item_image_features(new_item_image_path,
                                                                                           inference_model,
                                                                                           inference_saver,
                                                                                           inference_session)
            
            return extract_feature_result
        else:
            return 'Request not valid or model is not configured yet!'
    except:
        return 'Something went wrong!'


@app.route('/remove-item-image-features', methods=['POST'])
def remove_item_image_features():
    try:
        # Get item data from request
        item_image_path = request.get_json().get('item_image_path')

        if (item_image_path != None):
            # Extract all items's image feature
            remove_feature_result = extract_image_feature.remove_item_image_features(item_image_path)
            
            return remove_feature_result
        else:
            return 'Request not valid or model is not configured yet!'
    except:
        return 'Something went wrong!'


@app.route('/generate-outfit-recommendation', methods=['POST'])
def generate_outfit_recommendation():
    """
        Input:
            Object {
                query_items: Items Array[],
                query_keywords: String Array[],
            }

        Output:
            Recommendations Array [
                Item image paths Array[],
                ...
            ]
    """
    try:
        # Get data from request
        data = request.get_json()
        query_item_image_paths = list(data.get('query_item_image_paths'))
        query_keywords = list(data.get('query_keywords'))

        if (query_item_image_paths != None and
            query_keywords != None and
            inference_model_config != None and
            inference_model != None and
            inference_saver != None and
            inference_session != None):

            # Generate outfit recommendations
            outfit_generate_result = outfit_generation.run(query_item_image_paths,
                                                           query_keywords,
                                                           words,
                                                           inference_model_config,
                                                           inference_model,
                                                           inference_saver,
                                                           inference_session)

            return json.dumps(outfit_generate_result)
        else:
            return 'Request not valid or model is not configured yet!'
    except:
        return 'Something went wrong!'


@app.route('/predict-fashion-compatibility', methods=['POST'])
def predict_fashion_compatibility():
    """
        Input:
            Object {
                closet_items: Items Array[],
            }

        Output:
            Outfits Array [
                Outfit Object {
                    items: Items Array[]
                    score: float,
                },
                ...
            ]
    """
    try:
        # Get data from request
        data = request.get_json()
        items = list(data.get('items'))
        limit = int(data.get('limit'))

        if (items != None and limit != None):
            # 1. Find unique categories
            unique_categories = set(item['category_id'] for item in items)

            # 2. Generate all possible sets of items with unique category id
            # Group items by category
            grouped_items = {}
            for item in items:
                item_category = item['category_id']
                if item_category in grouped_items:
                    grouped_items[item_category].append(item)
                else:
                    grouped_items[item_category] = [item]
            
            # Generate all possible sets
            all_sets = []
            for combination in itertools.product(*(grouped_items[category_id] for category_id in unique_categories)):
                all_sets.append(list(combination))

            # 3. Predict fashion compatiblity of each outfit combinations
            if (inference_model_config != None and
                inference_model != None and
                inference_saver != None and
                inference_session != None):


                # Generate outfit recommendations
                fashion_compatibility_scores = fashion_compatibility.run(all_sets,
                                                                         inference_model_config,
                                                                         inference_model,
                                                                         inference_saver,
                                                                         inference_session)

                # 4. Return top 5 highest score outfit
                # Combine sets and fashion compatibility scores using zip
                combined_set_score = zip(all_sets, fashion_compatibility_scores)
                
                # Sort the combined data based on scores in descending order
                sorted_sets_by_score = sorted(combined_set_score, key=lambda x: x[1], reverse=True)
                
                # Create a new array with sets and scores
                result = []
                for item_set, score in sorted_sets_by_score[:limit]:
                    result.append({"items": item_set, "score": str(score)})

                return json.dumps(result)
        else:
            return 'Request not valid or model is not configured yet!'
    except:
        return 'Something went wrong!'

def built_inference_model():
    global inference_model_config, inference_model, inference_saver, inference_session

    # Build the inference graph
    inference_graph = tf.Graph()
    with inference_graph.as_default():
        inference_model_config = configuration.ModelConfig()
        inference_model_config.rnn_type = 'lstm'
        inference_model = polyvore_model.PolyvoreModel(inference_model_config, mode="inference")
        inference_model.build()
        inference_saver = tf.train.Saver()

    inference_graph.finalize()
    inference_session = tf.Session(graph=inference_graph)


if __name__ == "__main__":
    # Read word name in word dictionary
    words = open("data/final_word_dict.txt").read().splitlines()
    for i, w in enumerate(words):
        words[i] = w.split()[0]

    built_inference_model()

    # Check if image features file is exist
    if not os.path.isfile("data/features/image_features.pkl"):
        print("Feature file is not exist. Extracting all item image features...")
        extract_all_item_image_features()
    
    app.run(port=5000, host='0.0.0.0')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      