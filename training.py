# for garbage collection
import gc

# for warnings
import operator
import warnings

warnings.filterwarnings("ignore")

# utility libraries
import os
import copy
import tqdm
import numpy as np
import pandas as pd
import cv2, random, time, shutil, csv
import tensorflow as tf
import math
import Img

# keras libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.utils import to_categorical
from keras import backend as K

data_dir = './input'
data_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
class_names = sorted(data_df['breed'].unique())
print(f"No. of classes read - {len(class_names)}")
time.sleep(1)

images_list = sorted(os.listdir(os.path.join(data_dir, 'train')))
X = []
Y = []
i = 0

for image in tqdm.tqdm(images_list[:-1]):
    cls_name = data_df[data_df['id'] == image[:-4]].iloc[0, 1]
    cls_index = int(class_names.index(cls_name))

    image_path = os.path.join(data_dir, 'train', image)
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    res_image = cv2.resize(orig_image, Img.DEFAULT_RESOLUTION)
    X.append(res_image)
    Y.append(cls_index)
    i += 1

print(len(X), len(Y))
Xarr = np.array(X)
Yarr = np.array(Y).reshape(-1, 1)

del (X)
print(Xarr.shape, Yarr.shape)
gc.collect()

Yarr_hot = to_categorical(Y)
print(Xarr.shape, Yarr.shape)

# FEATURE EXTRACTION OF TRAINING ARRAYS
AUTO = tf.data.experimental.AUTOTUNE


def get_features(model_name, data_preprocessor, data):
    """
    Create a feature extractor to extract features from the data.
    Returns the extracted features and the feature extractor.
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


def get_valfeatures(model_name, data_preprocessor, data):
    '''
    Used for feature extraction of validation and testing.
    '''

    dataset = tf.data.Dataset.from_tensor_slices(data)

    ds = dataset.batch(64)

    input_size = data.shape[1:]
    # Prepare pipeline.
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)

    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)

    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    # Extract feature.
    feature_maps = feature_extractor.predict(ds, verbose=1)
    # print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


# RETURNING CONCATENATED FEATURES USING MODELS AND PREPROCESSORS
def get_concat_features(feat_func, models, preprocs, array):
    print(f"Beggining extraction with {feat_func.__name__}\n")
    feats_list = []

    for i in range(len(models)):
        print(f"\nStarting feature extraction with {models[i].__name__} using {preprocs[i].__name__}\n")
        # applying the above function and storing in list
        feats_list.append(feat_func(models[i], preprocs[i], array))

    # features concatenating
    final_feats = np.concatenate(feats_list, axis=-1)
    # memory saving
    del (feats_list, array)
    gc.collect()

    return final_feats


# Models and preprocessors

from keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_preprocessor = preprocess_input

from keras.applications.xception import Xception, preprocess_input

xception_preprocessor = preprocess_input

from keras.applications.nasnet import NASNetLarge, preprocess_input

nasnet_preprocessor = preprocess_input

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

inc_resnet_preprocessor = preprocess_input

models = [InceptionV3, InceptionResNetV2, Xception, ]
preprocs = [inception_preprocessor, inc_resnet_preprocessor,
            xception_preprocessor, ]

final_train_features = get_concat_features(get_features, models, preprocs, Xarr)

# del(x_train, )
gc.collect()
print('Final feature maps shape', final_train_features.shape)

from keras.callbacks import EarlyStopping

EarlyStop_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                                                   verbose=0)

my_callback = [EarlyStop_callback]

from sklearn.model_selection import StratifiedKFold

splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=10).split(final_train_features, Y))

trained_models = []
accuracy = []
losses = []

# Prepare And Train DNN model

for i, (train_idx, valid_idx) in enumerate(splits):
    print(f"\nStarting fold {i + 1}\n")
    x_train_fold = final_train_features[train_idx, :]
    y_train_fold = Yarr_hot[train_idx, :]
    x_val_fold = final_train_features[valid_idx]
    y_val_fold = Yarr_hot[valid_idx, :]

    dnn = keras.models.Sequential([
        InputLayer(final_train_features.shape[1:]),
        Dropout(0.7),
        Dense(120, activation='softmax')
    ])

    dnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print("Training...")
    # Train simple DNN on extracted features.
    h = dnn.fit(x_train_fold, y_train_fold,
                batch_size=128,
                epochs=80,
                verbose=0,
                validation_data=(x_val_fold, y_val_fold),
                callbacks=my_callback)  # max 95.07

    print("Evaluating model ...")
    model_res = dnn.evaluate(x_val_fold, y_val_fold)

    accuracy.append(model_res[1])
    losses.append(model_res[0])
    trained_models.append(dnn)

print(f"\nAvg Accuracy - {sum(accuracy) / len(accuracy)}")
print(f"\nAvg Loss - {sum(losses) / len(losses)}")

best_accuracy_index, _ = max(enumerate(accuracy), key=operator.itemgetter(1))

# serialize to JSON
json_file = trained_models[best_accuracy_index].to_json()
with open("model.json", "w") as file:
    file.write(json_file)
# serialize weights to HDF5
trained_models[best_accuracy_index].save_weights("model.h5")

# SAVING RAM

del (final_train_features, Y, Yarr_hot, Xarr)
gc.collect()
