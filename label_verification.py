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

def predict_and_verify(model, test_images, test_labels):
    predictions = model.predict(test_images)

    correct_count = 0  
    total_count = len(test_labels)  

    print("\nManuelle Überprüfung der Vorhersagen:")

    for i in range(5):  
        idx = random.randint(0, len(test_images) - 1)

        predicted_label = np.argmax(predictions[idx])

        actual_label = test_labels[idx]

        predicted_letter = chr(predicted_label + 65)  
        actual_letter = chr(actual_label + 65)  

        is_correct = predicted_label == actual_label

        print(f"Beispiel {i + 1}:")
        print(f"  Tatsächliches Label: {actual_letter}")  
        print(f"  Vorhergesagtes Label: {predicted_letter}")  
        print(f"  {'KORREKT' if is_correct else 'FALSCH'}")  
        print("-" * 30)

        if is_correct:
            correct_count += 1

    accuracy = correct_count / total_count * 100  

    print(f"\nGesamtgenauigkeit bei manueller Überprüfung: {accuracy:.2f}%")

predict_and_verify(model, test_images, test_labels)

model.save("dateiname.keras")  
print("Das Modell wurde unter 'dateiname.keras' gespeichert.")