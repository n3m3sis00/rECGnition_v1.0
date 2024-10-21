import os
import glob
import wfdb
import numpy as np 
import csv

5class_targets = {
    "L":0,
    "N":0,
    "R":0,
    "e":0,
    "j":0,
    "A":1,
    "J":1,
    "S":1,
    "a":1,
    "E":2,
    "V":2,
    "F":3,
    "P":4,
    "U":4,
    "f":4
}

all_tragets={
    "L":0,
    "N":1,
    "R":2,
    "e":3,
    "j":4,
    "A":5,
    "J":6,
    "S":7,
    "a":8,
    "E":9,
    "V":10,
    "F":11,
    "P":12,
    "U":13,
    "f":14
}

def load_mitbih(data_fol):
        paths = glob.glob(data_fol + '/*.atr')

        with open("data.csv", "w", newline='') as file:
            writer = csv.writer(file)
            for file in paths:
                signal, field = wfdb.rdsamp(file[0:-4], channels = [1])
                ann = wfdb.rdann(file[0:-4], 'atr')
                ann_index = ann.sample
                ann_value = ann.symbol
                
                comment = field["comments"][0].split(" ")
                age = comment[0]
                if comment[1] == "M":
                    sex = 1
                else:
                    sex = 0

                med = field["comments"][1]

                for counter, index in enumerate(ann_index):
                    if counter == 1: continue

                    left_index = index - 129
                    right_index = index + 130

                    seg_signal = [x[0] for x in signal[left_index : right_index]]
                    seg_signal.append(age)
                    seg_signal.append(sex)
                    seg_signal.append(med)
                    seg_signal.append(ann_value[counter])
                    seg_signal.append(str(file[-7:-4])+"_" + str(counter))

                    writer.writerow(seg_signal)

        return None


if __name__ == "__main__":

    load_mitbih("../mitdb")
