import pandas as pd
import os
import time
import tqdm
import cv2
import Img
import tensorflow as tf
import gc

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Lambda, Input

data_dir = './input'
data_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
class_names = sorted(data_df['breed'].unique())
print(f"No. of classes read - {len(class_names)}")
time.sleep(1)

images_list = sorted(os.listdir(os.path.join(data_dir, 'train')))
X = []
Y = []

for image in tqdm.tqdm(images_list[:-1]):
    cls_name = data_df[data_df['id'] == image[:-4]].iloc[0, 1]
    cls_index = int(class_names.index(cls_name))

    image_path = os.path.join(data_dir, 'train', image)
    orig_image = cv2.imread(image_path)
    res_image = cv2.resize(orig_image, Img.DEFAULT_RESOLUTION)
    X.append(orig_image)
    Y.append(cls_index)

# FEATURE EXTRACTION OF TRAINING ARRAYS
AUTO = tf.data.experimental.AUTOTUNE


def get_features(model_name, data_preprocessor, data):
    """
    1- Create a feature extractor to extract features from the data.
    2- Returns the extracted features and the feature extractor.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)

    def preprocess(x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.5)
        return x

    ds = dataset.map(preprocess, num_parallel_calls=AUTO).batch(64)

    input_size = data.shape[1:]

    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)

    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)

    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(ds, verbose=1)

    del (feature_extractor, base_model, preprocessor, dataset)
    gc.collect()

    return feature_maps
