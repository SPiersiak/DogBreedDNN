import pandas as pd
import os
import time
import tqdm
import cv2
import Img

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
