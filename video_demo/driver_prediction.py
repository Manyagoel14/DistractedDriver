import os
import json
import shutil
import pickle

import numpy as np
import pandas as pd

# Pillow fix for truncated image loading
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TensorFlow/Keras imports — updated for TF 2.13+ and Python 3.11
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# tqdm fix for Windows (notebook version often fails)
from tqdm import tqdm                         

PICKLE_DIR = f"./pickle_files"
BASE_MODEL_PATH = f"./model"
JSON_DIR = f"./json_files"

if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)
    
BEST_MODEL = "D:\DistractedDriver\model\cnn_batchwise_best_model.keras"
model = load_model(BEST_MODEL)

with open("D:\DistractedDriver\pickle_files\labels_list_cnn_batchwise.pkl","rb") as handle:
    labels_id = pickle.load(handle)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True 

# test_tensors = paths_to_tensor(data_test.iloc[:,0]).astype('float32')/255 - 0.5
# image_tensor = paths_to_tensor(image_path).astype('float32')/255 - 0.5

def predict_result(image_tensor):
    # below are for shallow deep non batch: 64 and shallow non batch: 224
    # feature_extractor = VGG16(include_top=False, input_shape=(224, 224,3), weights='imagenet')
    # test_features = feature_extractor.predict(image_tensor)
    # ypred_test = model.predict(test_features)
    ypred_test = model.predict(image_tensor,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)
    print(ypred_class)

    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    ypred_class = int(ypred_class)
    print(id_labels[ypred_class])

    class_name = dict()
    class_name["c0"] = "SAFE_DRIVING"
    class_name["c1"] = "TEXTING_RIGHT"
    class_name["c2"] = "TALKING_PHONE_RIGHT"
    class_name["c3"] = "TEXTING_LEFT"
    class_name["c4"] = "TALKING_PHONE_LEFT"
    class_name["c5"] = "OPERATING_RADIO"
    class_name["c6"] = "DRINKING"
    class_name["c7"] = "REACHING_BEHIND"
    class_name["c8"] = "HAIR_AND_MAKEUP"
    class_name["c9"] = "TALKING_TO_PASSENGER"


    with open(os.path.join(JSON_DIR,'class_name_map.json'),'w') as secret_input:
        json.dump(class_name,secret_input,indent=4,sort_keys=True)

    with open(os.path.join(JSON_DIR,'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
        label = info[id_labels[ypred_class]]
        print(label)
    
    return label
