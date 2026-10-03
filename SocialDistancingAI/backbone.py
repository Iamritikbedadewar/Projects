import glob, os, tarfile, urllib
import tensorflow as tf
from utils import label_map_util


def set_model(model_name, label_name):
    model_found = 0
    
    for file in glob.glob("*"):
        if file == model_name:
            model_found = 1
    
  
    model_name = model_name
    model_file = model_name + ".tar.gz"
    download_base = "http://download.tensorflow.org/models/object_detection/"

    path_to_ckpt = model_name + "/frozen_inference_graph.pb"
    

    path_to_labels = os.path.join("data", label_name)

    num_classes = 90

    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = os.path.join(model_dir, "saved_model")

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    return model, category_index

