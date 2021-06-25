import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def recognize_image(img_path):
    path = "input/"
    train_path = os.path.join(path, "train/*")
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    id2breed = {i: name for i, name in enumerate(breed)}

    ## Model
    model = tf.keras.models.load_model("model.h5")


    image = read_image(img_path, 224)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0]
    label_idx = np.argmax(pred)
    top3 = np.argsort(pred)[-4:][::-1]
    possible_breed = list()
    print(str(id2breed[top3[0]]).replace("_"," "))
    possible_breed.append(str(id2breed[top3[0]]).replace("_"," "))
    possible_breed.append(str(id2breed[top3[1]]).replace("_"," "))
    possible_breed.append(str(id2breed[top3[2]]).replace("_"," "))
    possible_breed.append(str(id2breed[top3[3]]).replace("_"," "))
    return str(id2breed[label_idx]).replace("_"," "), possible_breed

if __name__ == "__main__":
    print(recognize_image())

