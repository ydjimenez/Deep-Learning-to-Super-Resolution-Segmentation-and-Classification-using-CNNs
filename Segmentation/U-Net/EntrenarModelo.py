
from GeneradorImagen import generador_img
from RedUnet import Unet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

print("Versión de tensorflow:{}".format(tf.__version__))
print("GPU:{}".format(tf.test.gpu_device_name()))

validar = True
guardar_modelo = True
guardar_historial = True
DIM_ORDER = "channels_last"

# Parámetros entrenamiento
ruta_img_train = "Datos/Entrenamiento/RoiImg/"
ruta_mascara_train = "Datos/Entrenamiento/RoiMascara/"
tamanio_train = 628 
tamanio_lote = 4 
epocas = 40 
steps = tamanio_train/tamanio_lote 

# Parámetros validación
ruta_img_val = "Datos/Validacion/RoiImg/"
ruta_mascara_val = "Datos/Validacion/RoiMascara/"
tamanio_val = 156 
tamanio_lote_val = 4
steps_val = tamanio_val/tamanio_lote_val 
altura = 128
ancho = 128

guardar_modelo = "ModeloEntrenado/modelo_entrenado"

modelo = Unet(input_altura=altura, input_ancho=ancho)
modelo.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy']) 
generador_entrenamiento = generador_img(ruta_img_train, ruta_mascara_train, tamanio_lote_val, altura, ancho, DIM_ORDER)

if not validar:
    history = modelo.fit_generator(generador_entrenamiento, steps, epocas=epocas, verbose=1)

else:
    generador_validacion = generador_img(ruta_img_val, ruta_mascara_val, tamanio_lote_val, altura, ancho, DIM_ORDER)
    history = modelo.fit_generator(generador_entrenamiento, steps, validation_data=generador_validacion, validation_steps=steps_val, epochs=epocas, verbose=1) # metodo fit() -> Entrena el modelo para un número fijo de épocas (iteraciones en un conjunto de datos).

if guardar_modelo:
   modelo.save(guardar_modelo + ".h5")

if guardar_historial:

    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_csv("Archivos/Valores_exactitud_pérdida.csv", mode="w+", index=False, header=True)
 
    x = np.arange(1, epocas+1, 1)
    plt.figure(figsize= (30,5))
    plt.subplot(121)
    plt.plot(x, history.history['accuracy']) 
    plt.plot(x, history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['Train', 'test'], loc='upper left')
 
    plt.subplot(122)
    plt.plot(x, history.history['loss'])
    plt.plot(x, history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Archivos/Gráficas_exactitud_pérdida.png")