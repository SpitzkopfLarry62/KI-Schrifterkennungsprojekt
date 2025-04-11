import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import tkinter as tk
from tkinter import messagebox  # Import messagebox
from PIL import Image, ImageDraw
import os

MODEL_FILE = "mein_tf_model.h5"

def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        print("Loading preexisting model...")
        model = tf.keras.models.load_model(MODEL_FILE)
    else:
        print("No preexisting model found. Training a new model...")
       
model = load_or_train_model()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")

        self.canvas_width = 280  
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.model = tf.keras.models.load_model(MODEL_FILE)

    def paint(self, event):
        x, y = event.x, event.y
        brush_size = 10
        scaled_x = x * (28 / self.canvas_width)
        scaled_y = y * (28 / self.canvas_height)
        self.canvas.create_oval(x, y, x + brush_size, y + brush_size, fill='black')
        self.draw.ellipse([scaled_x, scaled_y, scaled_x + 1, scaled_y + 1], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        img_resized = np.array(self.image) / 255.0  
        img_resized = img_resized.reshape(1, 28, 28)

        predictions = self.model.predict(img_resized)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        print(f"Predicted Digit: {predicted_class}, Confidence: {confidence:.2f}")
        messagebox.showinfo("Prediction Result", f"Predicted Digit: {predicted_class}\nConfidence: {confidence:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
