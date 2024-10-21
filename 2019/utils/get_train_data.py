import pathlib
import os

import numpy as np
import tensorflow as tf

IMG_WIDTH =128
IMG_HEIGHT = 128


data_dir = pathlib.Path(".\\data\\flower_photos")
image_count = len(list(data_dir.glob('*\\*.jpg')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == CLASS_NAMES

def get_metadata(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  age = tf.string.split(parts[-1],"_")[-2]
  sex = tf.string.splig(parts[-1],"_")[-1][:-4]
  return tf.Constant([age, sex])

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  label = get_label(file_path)
  metadata = get_metadata(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  data = {"age_gen": metadata, "img": img}
  return data, label


labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.repeat()
  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_ds = prepare_for_training(labeled_ds)