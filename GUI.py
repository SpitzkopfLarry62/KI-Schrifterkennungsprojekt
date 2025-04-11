import tkinter as tk
from tkinter import Canvas, Label, Button
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ml_model = tf.keras.models.load_model("dateiname.keras")

CANVAS_DIMENSION = 200  
IMAGE_SCALE = 28

class CharacterRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Buchstaben-Erkennung")
        master.configure(bg='lightgray')

        master.resizable(False, False)

        self.heading_label = Label(master, text="Buchstaben-Erkennung", font=("Arial", 16, "bold"), bg='lightgray')
        self.heading_label.pack(pady=5)

        
        self.drawing_area = Canvas(master, width=CANVAS_DIMENSION, height=CANVAS_DIMENSION, bg='white', highlightthickness=2, highlightbackground='black')
        self.drawing_area.pack(pady=5)

        self.predict_button = Button(master, text="Erkennen", command=self.classify_character, bg='green', fg='white', font=("Arial", 10))
        self.predict_button.pack(pady=2)

        self.reset_button = Button(master, text="Löschen", command=self.reset_drawing, bg='red', fg='white', font=("Arial", 10))
        self.reset_button.pack(pady=2)

        self.result_label = Label(master, text="", font=("Arial", 12, "bold"), bg='lightgray')
        self.result_label.pack(pady=5)

        self.graph_frame = tk.Frame(master, bg='lightgray')
        self.graph_frame.pack(pady=5)

        self.drawing_area.bind("<B1-Motion>", self.draw)
        self.drawing_area.bind("<Button-1>", self.start_drawing)

        self.image_data = Image.new("L", (CANVAS_DIMENSION, CANVAS_DIMENSION), color=255)

    def start_drawing(self, event):
        self.previous_x, self.previous_y = event.x, event.y

    def draw(self, event):
        x, y = event.x, event.y
        self.drawing_area.create_line(self.previous_x, self.previous_y, x, y, fill="black", width=12, capstyle=tk.ROUND, smooth=tk.TRUE)
        image_pixels = self.image_data.load()
        for i in range(-6, 7):
            for j in range(-6, 7):
                if 0 <= x + i < CANVAS_DIMENSION and 0 <= y + j < CANVAS_DIMENSION:
                    image_pixels[x + i, y + j] = 0
        self.previous_x, self.previous_y = x, y

    def reset_drawing(self):
        self.drawing_area.delete("all")
        self.result_label.config(text="")
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        self.image_data = Image.new("L", (CANVAS_DIMENSION, CANVAS_DIMENSION), color=255)

    def classify_character(self):
        raw_image = np.array(self.image_data)
        resized_image = cv2.resize(raw_image, (IMAGE_SCALE, IMAGE_SCALE))
        normalized_image = 1 - (resized_image / 255.0)
        input_data = normalized_image.reshape(1, IMAGE_SCALE, IMAGE_SCALE)

        predictions = ml_model.predict(input_data)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        predicted_character = chr(65 + predicted_index)

        self.result_label.config(text=f"Vorhersage: {predicted_character} ({confidence:.1f}%)")

        self.plot_probabilities(predictions[0])

    def plot_probabilities(self, probabilities):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 2)) 
        alphabet = [chr(i + 65) for i in range(26)]
        ax.bar(alphabet, probabilities, color="blue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Wahrscheinlichkeit")
        ax.set_title("Klassifikations-Wahrscheinlichkeiten")
        fig.tight_layout()

        chart = FigureCanvasTkAgg(fig, self.graph_frame)
        chart.get_tk_widget().pack()
        chart.draw()

if __name__ == '__main__':
    root = tk.Tk()
    app = CharacterRecognitionApp(root)
    root.mainloop()