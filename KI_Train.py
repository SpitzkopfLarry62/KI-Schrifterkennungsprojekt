import os
import cv2
import numpy as np

DATASET_PATH = r"C:\Users\Baran62\Documents\KI\Tensa\a4_raster_Baran_Erdem\BigDataSet"
IMG_SIZE = 28  
OUTPUT_IMAGES = "images.npy"
OUTPUT_LABELS = "labels.npy"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Der Pfad '{DATASET_PATH}' existiert nicht!")

images = []
labels = []

for label_char in sorted(os.listdir(DATASET_PATH)):
    label_path = os.path.join(DATASET_PATH, label_char)
    
    if not os.path.isdir(label_path):  
        continue

    if 'A' <= label_char <= 'Z':  
        label_num = ord(label_char) - ord('A')
    else:
        print(f"Warnung: '{label_char}' ist kein gültiger Buchstabe! Überspringe...")
        continue

    for filename in os.listdir(label_path):
        file_path = os.path.join(label_path, filename)

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  

            if image is None:
                print(f"⚠ Warnung: Datei '{filename}' konnte nicht geladen werden. Überspringe...")
                continue  

            if image.shape != (IMG_SIZE, IMG_SIZE):
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            normalized_img = 1 - image / 255.0

            images.append(normalized_img)
            labels.append(label_num)

if not images:
    raise RuntimeError("Keine validen Bilder gefunden! Überprüfe den Datensatz.")

images_array = np.array(images, dtype=np.float32)  
labels_array = np.array(labels, dtype=np.int32)  

np.save(OUTPUT_IMAGES, images_array)
np.save(OUTPUT_LABELS, labels_array)

print(f"Daten gespeichert: {OUTPUT_IMAGES} ({images_array.shape}), {OUTPUT_LABELS} ({labels_array.shape})")
