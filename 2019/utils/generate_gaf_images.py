import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import cv2

folder_loc_gaf = "../data/gaf_data"
folder_loc_images = "../data/image_data"

def save_img_gaf(data, id, class_name):

    fig = plt.figure(frameon=False)
    plt.plot(data)
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    filename = os.path.join(folder_loc_images, class_name ,str(id) + ".png")
    fig.savefig(filename, bbox_inches='tight', pad_inches=0,dpi=200)
    plt.close()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    val = data.T*data - np.sqrt(np.absolute(np.ones(len(data)) - data.T * data)).T * np.sqrt(np.absolute(np.ones(len(data)) - data.T * data))
    fig = plt.figure(frameon=False)
    plt.imshow(val, cmap='rainbow', origin='lower')
    plt.xticks([]), plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    filename = os.path.join(folder_loc_gaf, class_name ,str(id) + ".png")
    fig.savefig(filename, bbox_inches='tight', pad_inches=0,dpi=200)
    plt.close()

    return None



if __name__ == "__main__":
    data = pd.read_csv("..\\data\\data.csv")
    data = data.values
    for i in range(len(data)):
        print(i)
        if data[i][-2] == "~" or data[i][-2] == "!" or data[i][-2] == '"' or data[i][-2] == "+" or data[i][-2] == "/" or data[i][-2] == "[" or data[i][-2] == "]" or data[i][-2] == "" or data[i][-2] == "|": continue
        else: save_img_gaf(data[i][:258], data[i][-1] + "_" + str(data[i][-5]) +"_" + str(int(data[i][-4])), data[i][-2]) 