import timeit
import pathlib
import os
from datetime import datetime


import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from models.architectures.final_model import getModel
from utils.write_evaluation import write_evaluation
from models.architectures.final_model import AttentionModule


BASE_RESULT_DIR = os.getcwd()
IMG_WIDTH = 64
IMG_HEIGHT = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
# TOTAL_SAMPLE = 102794
BATCH_SIZE = 16
EPOCHS=25
# TEST_SIZE = int((0.5 * TOTAL_SAMPLE)/BATCH_SIZE)
DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
CSV_PATH = os.path.join(BASE_RESULT_DIR , "results" , "train_" + DATE_TIME + ".csv")
CSV_PATH_TEST = os.path.join(BASE_RESULT_DIR , "results" , "test_" + DATE_TIME + ".csv")

file1 = open("test_path.txt","w")#write mode 
file1.write(CSV_PATH_TEST) 
file1.close()

data_dir = pathlib.Path("./data/gaf_data_all")
image_count = len(list(data_dir.glob('*/*.png')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == CLASS_NAMES

def get_metadata(file_path):
  # parts = tf.strings.split(file_path, os.path.sep)
  # age = tf.strings.split(parts[-1],"_")[-2]
  # sex = tf.strings.split(parts[-1],"_")[-1].numpy()[:-4]
  age = 69
  sex = 1
  return tf.constant([age, sex])

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  # i = 0
  # while i < TOTAL_SAMPLE:
  label = get_label(file_path)
  metadata = get_metadata(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  data = {"age_gen": metadata, "img": img}
  # yield data, label
  # i += 1
  return data, label




def prepare_for_training(dataset, epochs, cache=True, shuffle_buffer_size=1000):
  # dataset = tf.data.Dataset.from_generator(lambda:process_path(ds))

  if cache:
    if isinstance(cache, str):
      dataset = dataset.cache(cache)
    else:
      dataset = dataset.cache()
  dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
  dataset = dataset.repeat(epochs)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  # test_set = ds.take(TEST_SIZE)
  # train_set = ds.skip(TEST_SIZE)
  iterator = dataset.make_one_shot_iterator()
  while True:
      batch_features, batch_labels = iterator.get_next()
      yield batch_features, batch_labels

  # return dataset



labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# print(labeled_ds)
train_ds = prepare_for_training(labeled_ds, EPOCHS)



METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

model = getModel()

print("compiling model....")
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=METRICS
                  )
print("compilation completed....")


csv_logger = tf.keras.callbacks.CSVLogger(CSV_PATH)
        
print("Training started....")

#definig weights of each class 
## weight_of_class_0 = (1/class_0_count)*(total_count)/total_num_of_classes
# class_weights = {
#                     0: 0.097922,
#                     1: 2.89118,
#                     2: 32.15658,
#                     3: 9.18186,
#                     4: 38.15469,
#                     5: 1.033088,
#                     6: 49.09237,
#                     7: 88.72116,
#                     8: 223.1472,
#                     9: 460.24105,
#                     10: 0.912491,
#                     11: 69.47034,
#                     12: 7.498835,
#                     13: 0.975086,
#                 }

start = timeit.default_timer()
model.fit_generator(generator = train_ds,
           epochs=EPOCHS,
           steps_per_epoch=44,
           callbacks=[csv_logger],
          #  class_weight=class_weights
           )
stop = timeit.default_timer()
print("training finished")

model.save("model.h5")


print("------------------------------------------------->")
print('Time Taken to train: ', str((stop - start)/60) + " min")
print("------------------------------------------------->")

