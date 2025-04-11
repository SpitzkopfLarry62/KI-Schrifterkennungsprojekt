import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
import random

image_data = np.load("images.npy")  
label_data = np.load("labels.npy").astype('int32').flatten()

train_images, test_images, train_labels, test_labels = train_test_split(
    image_data, label_data, test_size=0.1, random_state=101)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.25))  
model.add(Dense(128, activation='relu'))  
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=12,  
          batch_size=32,  
          validation_data=(test_images, test_labels))

final_loss, final_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Testgenauigkeit: {final_accuracy * 100:.2f}%")

model.save("sequential_model_no_normalization.h5")
print("Das Modell wurde unter 'sequential_model_no_normalization.h5' gespeichert.")