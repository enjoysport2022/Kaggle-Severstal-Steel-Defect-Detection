#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用  "0,1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config=config)

# In[2]:

train_df = pd.read_csv('../input/train.csv', engine='python')
submission_df = pd.read_csv('../input/sample_submission.csv', engine='python')

# In[5]:


unique_test_images = submission_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[0]
).unique()


train_df['isNan'] = pd.isna(train_df['EncodedPixels'])
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[0]
)


train_nan_df = train_df.groupby(by='ImageId', axis=0).agg('sum')
train_nan_df.reset_index(inplace=True)
train_nan_df.rename(columns={'isNan': 'missingCount'}, inplace=True)
train_nan_df['missingCount'] = train_nan_df['missingCount'].astype(np.int32)
train_nan_df['allMissing'] = (train_nan_df['missingCount'] == 4).astype(int)

test_nan_df = pd.DataFrame(unique_test_images, columns=['ImageId'])


BATCH_SIZE = 4

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.1,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,
        rotation_range=10,
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.,
        validation_split=0.15
    )

def create_test_gen():
    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
        test_nan_df,
        directory='../input/test_images/',
        x_col='ImageId',
        class_mode=None,
        target_size=(1600, 256),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        train_nan_df, 
        directory='../input/train_images/',
        x_col='ImageId', 
        y_col='allMissing', 
        class_mode="other",
        target_size=(1600, 256),
        batch_size=BATCH_SIZE,
        subset=subset
    )

# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, 'training')
val_gen = create_flow(data_generator, 'validation')
test_gen = create_test_gen()


# In[14]:

def build_model():

    densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(1600, 256, 3))
    
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(),
        metrics=['accuracy']
    )
    
    return model


# In[15]:
model = build_model()
print(model.summary())


total_steps = train_nan_df.shape[0] / BATCH_SIZE

checkpoint = ModelCheckpoint(
    'remove_model.h5', 
    monitor='val_acc',
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    verbose=1,
    min_lr=1e-6
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=total_steps * 0.85,
    validation_data=val_gen,
    validation_steps=total_steps * 0.15,
    epochs=40,
    callbacks=[checkpoint, reduce_lr]
)


history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# ## Save results as CSV files

model.load_weights('../save_model/remove_model.h5')

y_test = model.predict_generator(
    test_gen,
    steps=len(test_gen),
    verbose=1
)
test_nan_df['allMissing'] = y_test

history_df.to_csv('../output/history.csv', index=False)
train_nan_df.to_csv('../output/train_missing_count.csv', index=False)
test_nan_df.to_csv('../output/test_missing_count.csv', index=False)


# nohup python -u predict-missing-masks.py > step01.log 2>&1 &