import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image, ImageDraw
import os


(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("1")
# Convert the numpy array to a PIL image
#image = Image.fromarray(x_train[0])

# Show Image, print Label 
#image.show()
#print(f"Label: {y_train[0]}")

# Normalize Pixel values
x_train, x_test=x_train/255, x_test/255

model=Sequential([
    Flatten(input_shape=(28,28)), # 784 Inputs (float 0-1)
    Dense(128, activation='relu'), # 1. Hidden Layer
    Dense(64, activation='relu'), # 2. Hidden Layer
    Dense(10, activation='softmax') # Output: 0 bis 9
])
print("1")
# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
print("1")
# Fit the model
model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))
print("1")
# Print the loss 
test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print(f"\nTest Accuracy: {test_acc}")

# Save the model:
model.save("mein_tf_model.h5")
print(f"Model saved as mein_tf_model.h5")