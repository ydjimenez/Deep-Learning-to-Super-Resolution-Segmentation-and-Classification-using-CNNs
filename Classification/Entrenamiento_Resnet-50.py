import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.initializers import glorot_uniform
import tensorflow as tf
import splitfolders

print("Versión de tensorflow:{}".format(tf.__version__))
print("GPU:{}".format(tf.test.gpu_device_name()))

"""
Dividir nuestro dataset en entrnamiento, validación y pruebas
"""
# input_folder = "D:\Archivos_Alexis\Trabajo_Proyecto_UTPL\Proyecto UTPL Clasificador FINAL\Clasificador Benigno Maligno _ GPU\Clasificador con tres daset_128,128_70%\Clasificador 5, 128,128, lote\Entrada_dataset"
# output = "D:\Archivos_Alexis\Trabajo_Proyecto_UTPL\Proyecto UTPL Clasificador FINAL\Clasificador Benigno Maligno _ GPU\Clasificador con tres daset_128,128_70%\Clasificador 5, 128,128, lote\data"
# splitfolders.ratio(input_folder, output, seed = 42, ratio = (.8, .1, .1)) #Semilla para muestreo aleatorio de las imagens, entonces tienes que mantener la semilla como en 42
# """"
# de lo contrario, puede obtener resultados diferentes, debido a una inicialización diferente dela aleatoridad, entonces tienes esta proporción, por lo que esta relación indica cuánto porcentaje de imagenes
# quieres mantener como en el entrenamiento entonces pruebas luego la validación, así que quiero mantener 80
# del conjunto de datos de mi entrenamiento 10 para validacion y 10 para prueba, asi que una vez ejecuta esta parte del codigo que vas.
# llegar estas tres carpetas de comercio de pruebas y validacion, dentro del directorio de salida dado dentro tu codigo y esto tendra todos los demas categorias de imagenes en su interior una vez que haya terminado de dividir su conjunto de datos
# se importaras capas de la funcion de proceso para resnet50
# """
# help(splitfolders.ratio)

"""
Brindamos la direccion de 2 carpetas que contienen las imagenes (una de las imagenes que 
serviran para entrenar la red y la otra para validar los datos)
"""
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

"""
Hiperparametros
"""
epocas = 60
ancho, altura = 128, 128
batch_size = 2
#clases = 2

"""
Preparamos nuestras imagenes
"""
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255) # Reescalamos los pixeles de la imagen entre 0 y 1 (esto es fundamental para mejorar el entrenamiento)

validacion_datagen = ImageDataGenerator(rescale=1. / 255) # Para la data de validacion solo reescalamos

"""
Va abrir y prepara toda la carpeta de entrenamiento 
"""
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, ancho),
    batch_size=batch_size,
    class_mode='categorical')

"""
Va abrir y prepara toda la carpeta de validacion
"""
imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, ancho),
    batch_size=batch_size,
    class_mode='categorical')

pasos_entrenamiento = imagen_entrenamiento.n//imagen_entrenamiento.batch_size # Cantidad de imagenes de entrenamiento dividido para el batch_size
pasos_validacion = imagen_validacion.n//imagen_validacion.batch_size # Cantidad de imagenes de validacion dividido para el batch_size

"""
Crear Red CNN
"""
#Implementación del bloque de identidad
def identity_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

#Implementación de bloque convolucional
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

#Implementacion de ResNet-50
def ResNet50(input_shape = (128, 128, 3), classes = 2):   

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)


    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

  
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = ResNet50(input_shape = (altura, ancho, 3), classes = 2)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

H=model.fit(
    imagen_entrenamiento,
    steps_per_epoch=pasos_entrenamiento,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion)

guardar_historial = True

if guardar_historial:

    history_df = pd.DataFrame.from_dict(H.history)
    history_df.to_csv("archivos/Valores_exactitud_pérdida.csv", mode="w+", index=False, header=True)
 
    x = np.arange(1, epocas+1, 1)
    plt.figure(figsize= (30,5))
    plt.subplot(121)
    plt.plot(x, H.history['accuracy']) 
    plt.plot(x, H.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['Train', 'val'], loc='upper left')
 
    plt.subplot(122)
    plt.plot(x, H.history['loss'])
    plt.plot(x, H.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.xticks(x)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("archivos/Gráficas_exactitud_pérdida.png")

target_dir = './model/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./model/modelResnet50_breast.h5')
model.save_weights('./model/pesosResnet50_breast.h5')
