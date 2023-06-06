import cv2
import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog

# Cargar el modelo pre-entrenado
model = load_model('modelo_genero_emocion.h5') #aqui colocas la direccion del modelo a utilizar

# Crear una lista con las etiquetas de las emociones
lista_emociones = ['Enojo', 'Felicidad', 'Sorpresa', 'Tristeza']

# Crear una lista con las etiquetas de los géneros
lista_generos = ['Hombre','Mujer']

# Abrir ventana de explorador de archivos y permitir al usuario seleccionar la imagen de su conjunto de pruebas
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imagenes", "*.jpg;*.png")])

# Leer la imagen y analizarla
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)


    preds = model.predict(roi)[0]
    emotion = lista_emociones[np.argmax(preds)]

    # Hacer la predicción del género

    genero = lista_generos[np.argmax(preds)]

    img2 = mpimg.imread(image_path)
    plt.imshow(img2)
    plt.text(0, -10, 'Genero: '+genero+ " Emocion: "+ emotion, fontsize=14, color='black')
    plt.show()
