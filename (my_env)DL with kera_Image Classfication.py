# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:38:42 2022

@author: EP
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#import data

(train_images, train_labels),(test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
print(train_images.shape)

#With 60,000 training samples, we scale the data between 0 and 1 and specify
#class name for easier handling
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#First 16 training samples are displayed next with their known labels.
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Build the network, compile, train it, and save it to disk
#This model have 128 nodes and output layer with 10 nodes

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)])

model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
model.save("fashion_model.h5")

#Load the model
model= tf.keras.models.load_model('fashion_model.h5')

#Evaluation
test_loss, test_acc= model.evaluate(test_images, test_labels, verbose=2)
print('/nTest accuracy:', test_acc)

#Prediction
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])
print(test_labels[0])

#For defineteness we look at one prediction visually using two helper functions

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
      
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
  
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
    
plot_value_array(i, predictions[i], test_labels)
plt.show()

img = test_images[1]
img = (np.expand_dims(img,0))

predictions_single = probability_model.predict(img)#img2 on oma
print(class_names[np.argmax(predictions_single[0])])
print("true=",class_names[test_labels[1]])