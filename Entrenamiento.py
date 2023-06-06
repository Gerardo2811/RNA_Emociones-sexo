import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import cv2

data_path = 'DATASET' #Indicas la ruta de tu carpeta con el dataset
emotion_list = ['Enojo', 'Felicidad', 'Sorpresa', 'Tristeza']# lsita de emociones detectables
gender_list = os.listdir(data_path)
num_classes = len(emotion_list) * len(gender_list)

# Cargar las imágenes y etiquetas de la carpeta de datos
x_train = []
y_train = []
for idx_gender, gender in enumerate(gender_list):
    for idx_emotion, emotion in enumerate(emotion_list):
        folder_path = os.path.join(data_path, gender, emotion)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, 0)  # Leer la imagen en escala de grises
            img = cv2.resize(img, (48, 48))  # Redimensionar la imagen
            x_train.append(img)
            y_train.append(idx_gender * len(emotion_list) + idx_emotion)

# Convertir los datos a arreglos NumPy
x_train = np.array(x_train)
y_train = np.array(y_train)

# Normalizar los datos de entrada
x_train = x_train.astype('float32') / 255.0

# Definir la arquitectura de la red neuronal
model3 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model3.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar la red neuronal
model3.fit(x_train, y_train, epochs=15)

# Guardar el modelo entrenado
model3.save('modelo_genero_emocion44.h5')