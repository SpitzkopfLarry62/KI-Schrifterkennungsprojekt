# 📝 Intelligente Buchstabenerkennung

Dieses Projekt nutzt moderne KI-Technologien, um handgeschriebene **Blockbuchstaben** (A–Z) zu erkennen. Mit einer benutzerfreundlichen Oberfläche können Buchstaben gezeichnet werden, die von einem neuronalen Netz analysiert und klassifiziert werden. Die zugrunde liegende Technologie basiert auf `TensorFlow`, `OpenCV` und `tkinter`.

---

## 📂 Projektübersicht

```
📦 Intelligente-Buchstabenerkennung
├── assets/
│   ├── eingabe_raster.png         # Vorlage für handgeschriebene Buchstaben
│   ├── buchstaben_bilder/         # Gelabelte Buchstabenbilder
├── training/
│   ├── training_data.npy          # Trainingsdaten
│   ├── labels.npy                 # Labels der Buchstaben
│   ├── train_model.py             # Trainingsskript
│   ├── evaluate_model.py          # Konfusionsmatrix und Modellanalyse
│   ├── export_model.py            # Speichert das Modell in .keras und .h5
├── gui/
│   ├── letter_recognition_gui.py  # Hauptanwendung mit GUI
├── pre_processing/
│   ├── extract_letters.py         # Extrahiert Buchstaben aus Rasterbildern
│   ├── prepare_data.py            # Skaliert und normalisiert Trainingsdaten
├── models/
│   ├── final_model.keras          # Trainiertes Modell
│   ├── final_model.h5             # Alternative Modellversion
├── screenshots/
│   ├── gui_example.png            # Screenshot der Anwendung
├── requirements.txt               # Liste der benötigten Bibliotheken
└── README.md                      # Projektbeschreibung
```

---

## 📋 Projektbeschreibung

Mit diesem Projekt kannst du handgeschriebene Buchstaben erkennen lassen. Hierbei werden Buchstaben auf einem Rasterblatt geschrieben, eingescannt und in Trainingsdaten umgewandelt. Diese Daten werden genutzt, um ein neuronales Netz zu trainieren, das dann in der GUI verwendet wird.

Die wichtigsten Funktionen:
- Zeichne Buchstaben in der grafischen Benutzeroberfläche.
- Lass die KI den Buchstaben erkennen und die Ergebnisse in einem Balkendiagramm anzeigen.
- Nutze vorgefertigte Skripte, um Daten einfach zu verarbeiten und das Modell zu trainieren oder zu testen.

---

## 🛠️ Installation

### Voraussetzungen
- **Python 3.8 oder neuer**
- Benötigte Bibliotheken:
  ```bash
  pip install tensorflow opencv-python numpy matplotlib pillow
  ```

### Installation
1. Klone das Repository:
   ```bash
   git clone https://github.com/SpitzkopfLarry62/Intelligente-Buchstabenerkennung.git
   cd Intelligente-Buchstabenerkennung
   ```
2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Nutzung

### 1️⃣ Buchstaben vorbereiten
- Drucke die Vorlage `eingabe_raster.png` aus und schreibe die Buchstaben (A-Z) darauf.
- Scanne das Blatt ein und speichere es als Bild.
- Nutze das Skript `extract_letters.py`, um die Buchstaben automatisch auszuschneiden:
  ```bash
  python pre_processing/extract_letters.py
  ```

### 2️⃣ Daten vorbereiten
- Skaliere und normalisiere die Buchstabenbilder mit `prepare_data.py`:
  ```bash
  python pre_processing/prepare_data.py
  ```

### 3️⃣ Modell trainieren
- Trainiere das Modell mit den vorbereiteten Daten:
  ```bash
  python training/train_model.py
  ```
- Das trainierte Modell wird im Ordner `models/` gespeichert.

### 4️⃣ Modell analysieren
- Überprüfe die Leistung des Modells und visualisiere die Konfusionsmatrix:
  ```bash
  python training/evaluate_model.py
  ```

### 5️⃣ GUI starten
- Starte die Buchstabenerkennung mit der grafischen Oberfläche:
  ```bash
  python gui/letter_recognition_gui.py
  ```

---

## 🖼️ GUI-Vorschau

![GUI Vorschau](screenshots/gui_example.png)

---

## 📊 Features

- **Zeichenfläche**: Zeichne Buchstaben direkt mit der Maus.
- **Klassifikation**: Das Modell erkennt den Buchstaben und zeigt die prozentuale Wahrscheinlichkeit an.
- **Balkendiagramm**: Visualisiert die Wahrscheinlichkeiten für alle Buchstaben (A–Z).
- **Datenverarbeitung**: Einfache Skripte zur Vorbereitung und Analyse von Trainingsdaten.

---

## 🔧 Erweiterungen

Hier sind einige Ideen, wie dieses Projekt weiterentwickelt werden kann:
- **Unterstützung für Ziffern**: Füge die Erkennung von Zahlen (0–9) hinzu.
- **Mehrsprachige Buchstaben**: Trainiere das Modell für andere Alphabete (z. B. griechisch, kyrillisch).
- **Handschriftenerkennung**: Erweitere die Funktionalität auf ganze Wörter oder Sätze.

---

## 👤 Autor

Erstellt von **[SpitzkopfLarry62](https://github.com/SpitzkopfLarry62)**. Dieses Projekt ist ein Beispiel für den Einsatz von KI in der Handschriftenerkennung.

---
