import csv
import os

import tensorflow as tf
import numpy as np


def write_evaluation(model, test_data, predictions, path):
    splited_path = os.path.split(path)
    head = splited_path[0]
    tail = "model_" + splited_path[1].split("_")[-2]+ "_" + splited_path[1].split("_")[-1][:-4] + ".png"
    model_img_loc = os.path.join(head,tail)
    tf.keras.utils.plot_model(model, model_img_loc, show_shapes=True)
    
    matrix = tf.math.confusion_matrix(test_data, predictions)
    print(matrix)

    with open(path, "w", newline='') as fileh:
        writer = csv.writer(fileh)
        for i in matrix:
            writer.writerow(i.numpy())