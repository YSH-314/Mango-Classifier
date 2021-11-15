# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:35:04 2021

@author: User
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

generator = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.2,
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
    )

testgenerator = ImageDataGenerator(rescale = 1./255,
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
    )

target_size = [300, 300]
batch_size = 64

train_dir = 'Train'
test_dir = 'Dev'
dftrain = pd.read_csv(r'train.csv')
dftest = pd.read_csv(r'dev.csv')
print(dftrain.shape, dftest.shape)
print(dftrain.head())

train_gen = generator.flow_from_dataframe(dataframe=dftrain, directory=train_dir, x_col="image_id", y_col="label",
                                          class_mode="categorical",
                                          target_size=target_size, 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          seed=101,
                                          subset='training')

val_gen = generator.flow_from_dataframe(dataframe=dftrain, directory=train_dir, x_col="image_id", y_col="label",
                                         class_mode="categorical",
                                         target_size=target_size, 
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         seed=101,
                                         subset='validation')

test_gen = testgenerator.flow_from_dataframe(dataframe=dftest, directory=test_dir, x_col="image_id", y_col="label",
                                         class_mode="categorical",
                                         target_size=target_size, 
                                         batch_size=1, 
                                         shuffle=False,
                                         )
print(train_gen.samples, val_gen.samples, test_gen.samples)
print(train_gen.class_indices)
# initializing label list and feeding in classes/indices
labels = [None]*len(train_gen.class_indices)

for item, indice in train_gen.class_indices.items():
    labels[indice] = item
    
model_mckp = keras.callbacks.ModelCheckpoint('incv3.h5',
                                             monitor='val_categorical_accuracy', 
                                             save_best_only=True, 
                                             mode='max')
base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, 
                                               weights='imagenet', 
                                               pooling='avg', 
                                               input_shape=target_size+[3])

model_1 = tf.keras.Sequential([
    base_model,
    layers.Dense(300, activation='relu'), 
    layers.Dense(3, activation='softmax')
])
model_1.summary()

adam = optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
rlr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, verbose=1, min_delta=0.0001)

model_1.compile(optimizer=adam,
                loss='categorical_crossentropy', 
                metrics=['categorical_accuracy'])
history = model_1.fit_generator(generator=train_gen,
                        steps_per_epoch = train_gen.samples//batch_size,
                        validation_data=val_gen,
                        validation_steps= val_gen.samples//batch_size,
                        epochs=30, 
                        callbacks=[rlr, model_mckp])

from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import itertools

def get_conf_matrix(model, data_dir, size, image_size):
    predictions =[]
    true_y = []
    for x,y in testgenerator.flow_from_dataframe(dataframe=dftest, directory=test_dir, x_col="image_id", y_col="label",
                                         class_mode="categorical",
                                         target_size=target_size, 
                                         batch_size=1, 
                                         shuffle=False,
                                         ):
        predprob = model.predict(x)
        # decoding one-hot
        prediction = np.argmax(predprob, axis=1)
        y = np.argmax(y, axis =1)
        
        predictions = np.concatenate((predictions, prediction))
        true_y = np.concatenate((true_y, y))
        if len(predictions) >=size:
            break
    matrix = confusion_matrix(true_y, predictions)

    return matrix

model = tf.keras.models.load_model('incv3.h5')
conf_matrix_inc = get_conf_matrix(model, test_dir, test_gen.samples, target_size)
print(conf_matrix_inc[:,:])
del model
K.clear_session()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=False`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(conf_matrix_inc, labels, title = "Test confusion matrix")

sum([conf_matrix_inc[i, i] for i in range(3)])/test_gen.samples
