import os
import random
import argparse
import pandas as pd
import numpy as np
import json
import math
from tensorflow.keras.applications import ResNet50

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,confusion_matrix
import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import matplotlib
from sklearn import preprocessing
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

import wandb
from wandb.keras import WandbCallback
wandb.login(key=os.environ['WANDB_KEY'])


parser = argparse.ArgumentParser(description='Process Arguments.')
parser.add_argument('--train_on', help='train on kaggle/colab')

args = parser.parse_args()


EXPERIMENT_VERSION = 2
hparams = {
    "epochs" : 60,
    "img_size" : 128,
    "lr" : 0.0001,
    "seed" : 127,
    "batch_size" : 64,
    "backbone" : "efficientnetb1",
    "optimizer" : "adam",
    "loss" : "categorical_crossentropy",
    "channels" : 1,
    'is fine tune' : "no",
    "Description" : "Train on all fold on Tpu",
    "folds" : 9
}
EXPERIMENT_NAME = 'mit-bih'


if args.train_on == "kaggle":
    from kaggle_datasets import KaggleDatasets
    GCS_DS_PATH = KaggleDatasets().get_gcs_path('mitbih-images-gaf-128x128')
    TRAIN_FILES = sorted(tf.io.gfile.glob(GCS_DS_PATH + '/data*.tfrec'))[:18]
    TEST_FILES = sorted(tf.io.gfile.glob(GCS_DS_PATH + '/data*.tfrec'))[18:]
    COUNT_FILE = GCS_DS_PATH + '/image_count.json'
elif args.train_on == "colab":
    GCS_DS_PATH = "/content/drive/MyDrive/Research papers/arrhythmia/CODE/input/gaf/mitbih-images-gaf-128x128"
    TRAIN_FILES = sorted(tf.io.gfile.glob(GCS_DS_PATH + '/data*.tfrec'))[:18]
    TEST_FILES = sorted(tf.io.gfile.glob(GCS_DS_PATH + '/data*.tfrec'))[18:]
    print(GCS_DS_PATH + '/image_count.json')
    COUNT_FILE = GCS_DS_PATH + '/image_count.json'
else:
    from kaggle_datasets import KaggleDatasets
    GCS_DS_PATH = KaggleDatasets().get_gcs_path('mitbih-images-gaf-128x128')
    TRAIN_FILES = sorted(tf.io.gfile.glob(GCS_DS_PATH + '/data*.tfrec'))[:18]
    TEST_FILES = sorted(tf.io.gfile.glob(GCS_DS_PATH + '/data*.tfrec'))[18:]
    COUNT_FILE = GCS_DS_PATH + '/image_count.json'


IMG_TRAIN_SHAPE = [hparams["img_size"],hparams["img_size"]]
STRATEGY = None
ACCURACY = []
CALLBACKS = [None]
NAME = ['/', "A", 'E', 'F', 'L', 'N', 'R', 'V', 'a', 'f', 'j']

with open(COUNT_FILE) as f:
  countdata = json.load(f)

t = 0
for _x in range(16):
    t += countdata[str(_x)]

TOTAL_TRAIN_IMG = t
TOTAL_TEST_IMG = countdata["8"] + countdata["9"]
BATCH_SIZE = hparams["batch_size"] # 16
EPOCHES = hparams["epochs"]
SEED = hparams["seed"]
FOLDS = 9

def seed_everything():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(a=SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_everything()


def getLoss(name=None):
    if name=="binary_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    else:
        return tf.keras.losses.CategoricalCrossentropy()

def getOptimizer(name = None, lr=0.0001):
    if name=="adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def get_cosine_schedule_with_warmup(lr = 0.00004, num_warmup_steps = 5 , num_training_steps = EPOCHES, num_cycles=0.5):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return lrfn

# lrfn = get_cosine_schedule_with_warmup()
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)
# yapl.config.CALLBACKS.append(lr_schedule)

#     weight_for_0 = (1 / yapl.config.NEG_SMAPLE)*(yapl.config.TOTAL_TRAIN_IMG)/2.0 
#     weight_for_1 = (1 / yapl.config.POS_SMAPLE)*(yapl.config.TOTAL_TRAIN_IMG)/2.0

#     class_weight = {0: weight_for_0, 1: weight_for_1}

def process_training_data(data_file):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        'target11': tf.io.FixedLenFeature([], tf.int64),
        'target5': tf.io.FixedLenFeature([], tf.int64),
        'target2': tf.io.FixedLenFeature([], tf.int64),
        'sex' : tf.io.FixedLenFeature([], tf.int64),
        'age' : tf.io.FixedLenFeature([], tf.int64),
    }
    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)
    img = tf.image.decode_jpeg(data['image'], channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    # img = tf.stack([img, img, img], axis = -1)
    img = tf.reshape(img, [*IMG_TRAIN_SHAPE, 3])

    age = tf.cast(data['age'], tf.int32)
    sex = tf.cast(data['sex'], tf.int32)
    tab_data = [tf.cast(tfeat, dtype = tf.float32) for tfeat in [age, sex]]
    tabular_data = tf.stack(tab_data)

    target11 = tf.one_hot(data['target11'], depth=11)
    # target5 = tf.one_hot(data['target5'], depth=5)
    # target2 = tf.one_hot(data['target2'], depth=2)

    return {'inp1' : img, 'inp2' : tabular_data}, {"target11" : target11 } 

def process_testing_data(data_file):
    LABELED_TFREC_FORMAT = {
        "image_id": tf.io.FixedLenFeature([], tf.string), 
        "image": tf.io.FixedLenFeature([], tf.string), 
        'target11': tf.io.FixedLenFeature([], tf.int64),
        'target5': tf.io.FixedLenFeature([], tf.int64),
        'target2': tf.io.FixedLenFeature([], tf.int64),
        'sex' : tf.io.FixedLenFeature([], tf.int64),
        'age' : tf.io.FixedLenFeature([], tf.int64),
    }
    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)
    img = tf.image.decode_jpeg(data['image'], channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    # img = tf.stack([img, img, img], axis = -1)
    img = tf.reshape(img, [*IMG_TRAIN_SHAPE, 3])

    age = tf.cast(data['age'], tf.int32)
    sex = tf.cast(data['sex'], tf.int32)
    tab_data = [tf.cast(tfeat, dtype = tf.float32) for tfeat in [age, sex]]
    tabular_data = tf.stack(tab_data)

    target11 = tf.one_hot(data['target11'], depth=11)
    # target5 = tf.one_hot(data['target5'], depth=5)
    # target2 = tf.one_hot(data['target2'], depth=2)
    image_id = data["image_id"]

    return {'inp1' : img, 'inp2' : tabular_data}, {"target11" : target11, "image_id":  data['image_id']} 

def multimodeleffbx(backbone="efficientnetb0"):
    inp1  = tf.keras.layers.Input(shape = (*IMG_TRAIN_SHAPE, 3), name='inp1')
    inp2  = tf.keras.layers.Input(shape = (2,), name='inp2')
    if backbone == "efficientnetb0":
        eff = efn.EfficientNetB0(
                            weights=None,
                            include_top=False,
                            input_shape = (*IMG_TRAIN_SHAPE, 3)
                        )
    if backbone == "efficientnetb1":
        eff = efn.EfficientNetB1(
                            weights=None,
                            include_top=False,
                            input_shape = (*IMG_TRAIN_SHAPE, 3)
                        )
    
    x1    = eff(inp1)
    x1    = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x1    = tf.keras.layers.Dropout(0.1)(x1)
    
    x2    = tf.keras.layers.Dense(8)(inp2) ##more layers to come after that

    x     = tf.keras.layers.concatenate([x1, x2])
    x     = tf.keras.layers.Dropout(0.15)(x)

    output11     = tf.keras.layers.Dense(11, activation='sigmoid', name='target11')(x)
    # output5     = tf.keras.layers.Dense(5, activation='sigmoid', name='target5')(x)
    # output2     = tf.keras.layers.Dense(2, activation='sigmoid', name='target2')(x)

    model = tf.keras.models.Model(inputs = [inp1, inp2], outputs = [output11])

    return model


def fitengine(model, traindataset, valdataset = None, istraining = True, ACCURACY= ACCURACY, CONFIG=hparams):
    model.compile(
        optimizer   =  getOptimizer(name=CONFIG["optimizer"], lr=CONFIG['lr']), 
        loss        =  getLoss(name=CONFIG['loss']), 
        metrics     =  ACCURACY
    )

    history = model.fit(
                traindataset, 
                epochs            =   CONFIG['epochs'], 
                steps_per_epoch   =   TOTAL_TRAIN_IMG//CONFIG['batch_size'],
                callbacks         =   CALLBACKS,
                validation_data   =   valdataset,
                verbose           =   1
            )

    return history


fold = 0
def runagent():    
    with wandb.init(
        dir = "/root",
        allow_val_change = True,
        group = "sweep_{}".format(EXPERIMENT_VERSION),
    ) as _run:
#         CONFIG = {'backbone': 'efficientnetb1', 'batch_size': 8, 'epochs': 20, 'loss': 'binary_crossentropy', 'lr': 0.0008839986474537997, 'optimizer': 'adam', 'seed': 235}
        CONFIG = dict(wandb.config)
        print("Config", CONFIG)
        ## Setup TPUS
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            tpu = None

        if tpu:
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            STRATEGY = tf.distribute.experimental.TPUStrategy(tpu)
        else:
            STRATEGY = tf.distribute.get_strategy() 
            
        print(STRATEGY)
        ACCURACY = []
        # SETUP ACCURACY METRICS
        if STRATEGY is not None:
            with STRATEGY.scope():
                ACCURACY.extend([
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Recall(name='sensitivity')
                ])
        else: 
            ACCURACY.extend([
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Recall(name='sensitivity')
            ])

        
        if STRATEGY:
            CONFIG["batch_size"] = 8 * CONFIG["batch_size"]
            CONFIG["lr"] = 8 * CONFIG["lr"]
            
        print("Config 2", CONFIG)

        if STRATEGY is not None:
            with STRATEGY.scope():
                model = multimodeleffbx(backbone = CONFIG["backbone"])
        else: 
            model = multimodeleffbx(backbone = CONFIG["backbone"])


        CALLBACKS[0] = tf.keras.callbacks.ModelCheckpoint(
                                        'fold-%i.h5'%(fold+1), monitor='loss', verbose=1, save_best_only=True,
                                        save_weights_only=True, mode='min', save_freq='epoch')
        
        CALLBACKS.append(WandbCallback())
        
        first_run = False
        # training data
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        train_dataset = (
            tf.data.TFRecordDataset(
                TRAIN_FILES[:-2],
                num_parallel_reads=tf.data.experimental.AUTOTUNE
            ).with_options(
                ignore_order
            ).map(
                process_training_data,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).repeat(
            ).shuffle(
                CONFIG['seed']
            ).batch(
                CONFIG['batch_size']
            ).prefetch(
                tf.data.experimental.AUTOTUNE
            )
        )
        
        ignore_order = tf.data.Options()
        val_dataset = (
            tf.data.TFRecordDataset(
                TRAIN_FILES[-2:],
                num_parallel_reads=tf.data.experimental.AUTOTUNE
            ).with_options(
                ignore_order
            ).map(
                process_training_data,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).batch(
                CONFIG['batch_size']
            ).prefetch(
                tf.data.experimental.AUTOTUNE
            )
        )

        ignore_order = tf.data.Options()
        test_dataset = (
            tf.data.TFRecordDataset(
                TEST_FILES,  
                num_parallel_reads=tf.data.experimental.AUTOTUNE
            ).with_options(
                ignore_order
            ).map(
                process_testing_data,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).batch(
                CONFIG['batch_size'] *  4
            ).prefetch(
                tf.data.experimental.AUTOTUNE
            )
        )


        

        print("##"*30)
        print("#### FOLD {} ".format(fold+1))
        
        
        history = fitengine(model, train_dataset, valdataset=val_dataset, ACCURACY = ACCURACY, CONFIG=CONFIG); #trining model
        # HISTORY['fold_{}'.format(fold+1)] = history #storing history

        print('##'*30)

        

        STEPS = (TOTAL_TEST_IMG)//(CONFIG['batch_size']*4) + 1

        with STRATEGY.scope():
            model = multimodeleffbx(backbone=CONFIG['backbone'])
            
        model.load_weights('fold-%i.h5'%(fold+1))
        test_imgs = test_dataset.map(lambda data, ids: data)
        img_labels_ds = test_dataset.map(lambda data, ids: ids).unbatch()
        pred = model.predict(test_imgs,steps = int(STEPS), verbose=1)
        # # pred = pred.squeeze()
        test_labels = next(iter(img_labels_ds.batch(int(TOTAL_TEST_IMG) + 1)))
        pd.DataFrame({
                'image_id'  : test_labels["image_id"].numpy(), 
                'actual'  : np.argmax(test_labels["target11"].numpy(), axis=1), 
                'predicted'      : np.argmax(pred, axis=1)
                }).to_csv('prediction_{}.csv'.format(fold + 1), index=False)   

        df = pd.read_csv("prediction_{}.csv".format(fold + 1))

        # overall_gt.append(test_labels["11class"])
        # overall_pred.append(pred[0])

        
        harvest = confusion_matrix(df['actual'], df['predicted'])
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(harvest)
        ax.set_xticks(np.arange(len(NAME)))
        ax.set_yticks(np.arange(len(NAME)))
        ax.set_xticklabels(NAME)
        ax.set_yticklabels(NAME)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        for i in range(len(NAME)):
            for j in range(len(NAME)):
                text = ax.text(j, i, harvest[i, j],
                            ha="center", va="center", color="w")

        fig.tight_layout()
        wandb.log({"confusion matrix": fig})

        from sklearn.metrics import classification_report
        target_names = NAME
        x_ = classification_report(df['actual'], df['predicted'], target_names=target_names, output_dict = True)


        data = []
        for x in target_names:
            x__ = [x] + list(x_[x].values())
            data.append(x__)
        wandb.log({"data" : wandb.Table(data=data, columns=[""] + list(x_[target_names[0]].keys()))})
        wandb.save("prediction_{}.csv".format(fold  + 1 ))

        print(x_)


count = 20 # number of runs to execute
wandb.agent("yh85hhai", function=runagent, count=count, project="ECG")