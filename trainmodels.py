# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:43:08 2022

@author: Elijah Gallagher
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def train():
    #this function is to prep and save the three main algorithems.
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    
    #standard conv network
    model = None
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    model.save('conv_CIF10.model')
    
    
    #overfitted conv network
    model = None
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=15, 
                        validation_data=(test_images, test_labels))
    model.save('overfitted_conv_CIF10.model')
    
    
    #basic forward flowing nn
    NNM = None
    NNM = tf.keras.models.Sequential()
    NNM.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    NNM.add(tf.keras.layers.Dense(4096, activation='relu'))
    NNM.add(tf.keras.layers.Dense(2048, activation='relu'))
    NNM.add(tf.keras.layers.Dense(512, activation='relu'))
    NNM.add(tf.keras.layers.Dense(256, activation='relu'))
    NNM.add(tf.keras.layers.Dense(128, activation='relu'))
    NNM.add(tf.keras.layers.Dense(32, activation='relu'))
    NNM.add(tf.keras.layers.Dense(10, activation='softmax'))
    NNM.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    NNM.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
    NNM.save('basic_nn_CIF10.model')
    
    #overfitten basic nn
    NNM = None
    NNM = tf.keras.models.Sequential()
    NNM.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    NNM.add(tf.keras.layers.Dense(1024, activation='relu'))
    NNM.add(tf.keras.layers.Dense(512, activation='relu'))
    NNM.add(tf.keras.layers.Dense(256, activation='relu'))
    NNM.add(tf.keras.layers.Dense(128, activation='relu'))
    NNM.add(tf.keras.layers.Dense(64, activation='relu'))
    NNM.add(tf.keras.layers.Dense(32, activation='relu'))
    NNM.add(tf.keras.layers.Dense(10, activation='softmax'))
    NNM.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    NNM.fit(train_images, train_labels, epochs=15, 
                        validation_data=(test_images, test_labels))
    
    NNM.save('overfitted_basic_nn_CIF10.model')
   
    
if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    