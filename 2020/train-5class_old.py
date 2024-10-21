import yapl
import os
import re
import random
import pandas as pd
import numpy as np
import math
from tensorflow.keras.applications import ResNet50

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,confusion_matrix
import tensorflow as tf
import tensorflow.keras.backend as K
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn


import matplotlib

from sklearn import preprocessing

from yapl.config.config import Config
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

import wandb
from wandb.keras import WandbCallback
wandb.login(key=os.environ['WANDB_KEY'])

EXPERIMENT_VERSION = 11

hparams = {
    "epochs" : 20,
    "img_size" : 224,
    "lr" : 0.0001,
    "seed" : 127,
    "batch_size" : 64,
    "backbone" : "efficientnetb4",
    "optimizer" : "adam",
    "loss" : "categorical_crossentropy",
    "class_weights" : {0: 3, 1: 0.55},
    "channels" : 1,
    'is fine tune' : "no",
    "Description" : "With Simple Model"
}

class configSetup(Config):
    def __init__(self):
        super().__init__()
        self.DO_VAL_SPLIT = True
        self.EXPERIMENT_NAME = 'mit-bih'
        self.GCS_DS_PATH = KaggleDatasets().get_gcs_path('5-class-mitbih')
        self.TRAIN_FILES = tf.io.gfile.glob(self.GCS_DS_PATH + '/test*.tfrec')
        self.TEST_FILES = tf.io.gfile.glob(self.GCS_DS_PATH + '/train*.tfrec')
        self.TRAIN_CSV = '../input/data-mini.csv'
                
        
        self.TOTAL_TRAIN_IMG = 0
        
        self.IMG_TRAIN_SHAPE = [hparams["img_size"],hparams["img_size"]]
        self.DO_FINETUNE = True
        
        self.BATCH_SIZE = hparams["batch_size"] # 16
        self.EPOCHES = hparams["epochs"]
        self.SEED = hparams["seed"]
        self.FOLDS = 1
        
        self.LOSS = tf.keras.losses.CategoricalCrossentropy()
        self.OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=hparams["lr"])
        self.ACCURACY = []
        self.CALLBACKS = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', restore_best_weights=True,  mode = 'min', patience = 10, verbose = 1, min_delta = 0.001,
            )
        ]
        
        self.STRATEGY = None

configSetup().make_global()

yapl.backend = 'tf'

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    yapl.config.STRATEGY = strategy
    yapl.config.BATCH_SIZE = 32 * strategy.num_replicas_in_sync 
else:
    strategy = tf.distribute.get_strategy() 

#configuration
yapl.config.TOTAL_TRAIN_IMG = len(pd.read_csv(yapl.config.TRAIN_CSV))/2



def seed_everything():
    np.random.seed(yapl.config.SEED)
    tf.random.set_seed(yapl.config.SEED)
    random.seed(a=yapl.config.SEED)
    os.environ['PYTHONHASHSEED'] = str(yapl.config.SEED)

seed_everything()

def process_training_data(data_file):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "target": tf.io.FixedLenFeature([], tf.int64),
        'sex' : tf.io.FixedLenFeature([], tf.int64),
        'age' : tf.io.FixedLenFeature([], tf.int64),
    }
    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)
    img = tf.image.decode_jpeg(data['image'], channels=1)
    img = tf.cast(img, tf.float32) / 255.0
    # img = tf.stack([img, img, img], axis = -1)
    img = tf.reshape(img, [*yapl.config.IMG_TRAIN_SHAPE, 1])

    age = tf.cast(data['age'], tf.int32)
    sex = tf.cast(data['sex'], tf.int32)
    tab_data = [tf.cast(tfeat, dtype = tf.float32) for tfeat in [age, sex]]
    tabular_data = tf.stack(tab_data)

    label = tf.one_hot(data['target'], depth=5)

    return {'inp1' : img, 'inp2' : tabular_data}, label

def get_cosine_schedule_with_warmup(lr = 0.00004, num_warmup_steps = 5 , num_training_steps = yapl.config.EPOCHES, num_cycles=0.5):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return lrfn

# lrfn = get_cosine_schedule_with_warmup()
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)
# yapl.config.CALLBACKS.append(lr_schedule)

def multimodeleffbx():
    inp1  = tf.keras.layers.Input(shape = (*yapl.config.IMG_TRAIN_SHAPE, 1), name='inp1')
    inp2  = tf.keras.layers.Input(shape = (2,), name='inp2')
    effb3 = efn.EfficientNetB0(
                        weights=None,
                        include_top=False,
                        input_shape = (*yapl.config.IMG_TRAIN_SHAPE, 1)
                    )
    
    x1    = effb3(inp1)
    x1    = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x1    = tf.keras.layers.Dropout(0.1)(x1)
    
    x2    = tf.keras.layers.Dense(8)(inp2) ##more layers to come after that

    x     = tf.keras.layers.concatenate([x1, x2])
    x     = tf.keras.layers.Dropout(0.15)(x)
    x     = tf.keras.layers.Dense(5, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs = [inp1, inp2], outputs = [x])

    return model

def fitengine(model, traindataset, valdataset = None, istraining = True):
    model.compile(
        optimizer   =  yapl.config.OPTIMIZER, 
        loss        =  yapl.config.LOSS, 
        metrics     =  yapl.config.ACCURACY
    )
#     weight_for_0 = (1 / yapl.config.NEG_SMAPLE)*(yapl.config.TOTAL_TRAIN_IMG)/2.0 
#     weight_for_1 = (1 / yapl.config.POS_SMAPLE)*(yapl.config.TOTAL_TRAIN_IMG)/2.0

#     class_weight = {0: weight_for_0, 1: weight_for_1}
    history = model.fit(
                traindataset, 
                epochs            =   yapl.config.EPOCHES, 
                steps_per_epoch   =   yapl.config.TOTAL_TRAIN_IMG//yapl.config.BATCH_SIZE,
                callbacks         =   yapl.config.CALLBACKS,
                validation_data   =   valdataset,
#                 validation_steps  =   yapl.config.TOTAL_VAL_IMG//yapl.config.BATCH_SIZE,
                verbose           =   1
            )

    return history


MODELS = {}
HISTORY = {}

print('##'*30)
print("#### IMAGE SIZE {} ".format((224, 224)))
print("#### BATCH SIZE {} ".format(yapl.config.BATCH_SIZE))
print("#### TRAINING RECORDS {} ".format(1))
print('##'*30)
print("\n\n")


ignore_order = tf.data.Options()
test_dataset = (
    tf.data.TFRecordDataset(
        yapl.config.TEST_FILES,  
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    ).with_options(
        ignore_order
    ).map(
        process_training_data,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(
        yapl.config.BATCH_SIZE *  4
    ).prefetch(
        tf.data.experimental.AUTOTUNE
    )
)

overall_pred = []
overall_gt = []


# skf = KFold(n_splits=yapl.config.FOLDS,shuffle=True,random_state=yapl.config.SEED)
# FOLDS_DICT = {}
# for fold,(idxT,idxV) in enumerate(skf.split(np.arange(len(yapl.config.TRAIN_FILES)))):
#     FOLDS_DICT['fold_{}'.format(fold+1)] = {
#                                             "trainfiles" : [yapl.config.TRAIN_FILES[x] for x in idxT],
#                                             "valfiles"   : [yapl.config.TRAIN_FILES[x] for x in idxV]
#                                             }

for fold in range(yapl.config.FOLDS):
    # print(FOLDS_DICT['fold_{}'.format(fold+1)])
    
    run_ = wandb.init(
        project="ECG", 
        reinit=True,
        dir = "/root",
        allow_val_change = True,
        config=hparams,
        group = "5_class_{}".format(EXPERIMENT_VERSION),
        name="5_class_{}_{}".format(EXPERIMENT_VERSION, fold),
    )
    
    
    yapl.config.CALLBACKS[0] = tf.keras.callbacks.ModelCheckpoint(
                                    'fold-%i.h5'%(fold+1), monitor='loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='min', save_freq='epoch')
    
    yapl.config.CALLBACKS.append(WandbCallback())
            
    # training data
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    train_dataset = (
        tf.data.TFRecordDataset(
            # FOLDS_DICT['fold_{}'.format(fold+1)]['trainfiles'], 
            yapl.config.TRAIN_FILES,
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        ).with_options(
            ignore_order
        ).map(
            process_training_data,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).repeat(
        ).shuffle(
            yapl.config.SEED
        ).batch(
            yapl.config.BATCH_SIZE
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
    )
    
    ignore_order = tf.data.Options()
    val_dataset = (
        tf.data.TFRecordDataset(
            # FOLDS_DICT['fold_{}'.format(fold+1)]['valfiles'],  
            yapl.config.TEST_FILES,
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        ).with_options(
            ignore_order
        ).map(
            process_training_data,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(
            yapl.config.BATCH_SIZE
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
    )


    if yapl.config.STRATEGY is not None:
        with strategy.scope():
            yapl.config.ACCURACY = []
            x2 = tf.keras.metrics.Precision(name='precision')
            x3 = tf.keras.metrics.BinaryAccuracy(name='accuracy')
            x4 = tf.keras.metrics.Recall(name='sensitivity')
            
            yapl.config.ACCURACY.append(x2)
            yapl.config.ACCURACY.append(x3)
            yapl.config.ACCURACY.append(x4)

            model = multimodeleffbx()
    else:
        yapl.config.ACCURACY = []
        x2 = tf.keras.metrics.Precision(name='precision')
        x3 = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        x4 = tf.keras.metrics.Recall(name='sensitivity')

        yapl.config.ACCURACY.append(x2)
        yapl.config.ACCURACY.append(x3)
        yapl.config.ACCURACY.append(x4)
        
        model = multimodeleffbx()

    print("##"*30)
    print("#### FOLD {} ".format(fold+1))
    
    
    history = fitengine(model, train_dataset, valdataset=val_dataset); #trining model
    # HISTORY['fold_{}'.format(fold+1)] = history #storing history

    print('##'*30)

    

    STEPS = (yapl.config.TOTAL_TRAIN_IMG)//(yapl.config.BATCH_SIZE*4) + 1

    # with strategy.scope():
    #     model = multimodeleffbx()
        
    # model.load_weights("fold-1.h5")
    test_imgs = test_dataset.map(lambda data, ids: data)
    img_labels_ds = test_dataset.map(lambda data, ids: ids).unbatch()
    pred = model.predict(test_imgs,steps = int(STEPS), verbose=1)
    # pred = pred.squeeze()
    test_labels = next(iter(img_labels_ds.batch(int(yapl.config.TOTAL_TRAIN_IMG) + 1))).numpy()
    print(pred.shape)
    print(test_labels.shape)
    pd.DataFrame({
            'actual'  : np.argmax(test_labels, axis=1), 
            'predicted'      : np.argmax(pred, axis=1)
            }).to_csv('prediction_{}.csv'.format(fold), index=False)   

    df = pd.read_csv("prediction_{}.csv".format(fold))

    overall_gt.append(test_labels)
    overall_pred.append(pred)

    NAME = ['N', 'SEB', 'VEB', "F", "Q"]
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
    wandb.save("prediction_{}.csv".format(fold))

    print(x_)
    run_.finish()
## complete prediciton


# run_ = wandb.init(
#         project="ECG", 
#         reinit=True,
#         dir = "/root",
#         allow_val_change = True,
#         config=hparams,
#         group = "5_class_{}".format(EXPERIMENT_VERSION),
#         name="5_class_{}_{}".format(EXPERIMENT_VERSION, "ens"),
#     )

# overall_gt = np.argmax(np.mean(np.array(overall_gt), axis=0), axis = 1)
# overall_pred = np.argmax(np.mean(np.array(overall_pred), axis = 0), axis = 1)

# print(overall_gt[0])
# print(len(overall_gt))
# print(len(overall_pred))

# pd.DataFrame({
#             'actual'  : overall_gt, 
#             'predicted'      : overall_pred
#             }).to_csv('prediction.csv', index=False)   

# df = pd.read_csv("prediction.csv")
# NAME = ['N', 'SEB', 'VEB', "F", "Q"]
# harvest = confusion_matrix(df['actual'], df['predicted'])
# fig, ax = plt.subplots(figsize=(8,8))
# im = ax.imshow(harvest)
# ax.set_xticks(np.arange(len(NAME)))
# ax.set_yticks(np.arange(len(NAME)))
# ax.set_xticklabels(NAME)
# ax.set_yticklabels(NAME)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

# for i in range(len(NAME)):
#     for j in range(len(NAME)):
#         text = ax.text(j, i, harvest[i, j],
#                     ha="center", va="center", color="w")

# fig.tight_layout()
# wandb.log({"confusion matrix": fig})

# from sklearn.metrics import classification_report
# target_names = NAME
# x_ = classification_report(df['actual'], df['predicted'], target_names=target_names, output_dict = True)


# data = []
# for x in target_names:
#     x__ = [x] + list(x_[x].values())
#     data.append(x__)
# wandb.log({"data" : wandb.Table(data=data, columns=[""] + list(x_[target_names[0]].keys()))})
# wandb.save("prediction.csv")

# print(x_)
# run_.finish()
