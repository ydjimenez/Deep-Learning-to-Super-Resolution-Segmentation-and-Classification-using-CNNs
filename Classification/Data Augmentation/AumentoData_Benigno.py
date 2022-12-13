
#Importamos las librerias necesarias aumento de la data
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, image, img_to_array

print("Versión de tensorflow:{}".format(tf.__version__))
print("GPU:{}".format(tf.test.gpu_device_name()))

carpeta_benignoAum = 'benigno'  #Crear carpeta guardar aumento imagenes benigno

num_imagsBenignoMalignoAum = 6 #Defimos el número de imagenes que se van a generar por cada imagen original (benigno Y maligno)

"""
Creamos la carpeta en el directorio en caso de no tenerla creada
"""
try:
    os.mkdir(carpeta_benignoAum)
except:
    print("")

"""
Configuramos las transformaciones que se le aplican a cada imagen para generara nuevas imagenes
"""

"""
Aumento Imagenes Benignas
"""
aumento_imagenesB = ImageDataGenerator( #Llamamos el paquete que se importo de keras, mismo que nos permite definir o configurar el generador de imagenes
    rotation_range = 30,       # Rotación: se configura en grados, se rotara la imagen a 20 grados
    zoom_range = 0.3,          #Zoom: esta configurado como una fracción, en este caso un zoom más o menos del 30% de la imagen
    width_shift_range = 0.2,   #Desplazar la imagen a lo largo de la dimensión del ancho, en este caso realizadesplazamiento del 20% de ancho de la imagen
    height_shift_range = 0.2,  #Desplazar la imagen a lo largo de la dimensión de la altura, en este caso realizadesplazamiento del 20% de ancho de la imagen
    horizontal_flip = True,   #Inversión horizontal de la imagen, es decir efecto espejo 
    vertical_flip = True,     #Inversión vertical de la imagen, es decir  efecto espejo 
    fill_mode='reflect'       #Los puntos fuera de los límites de la entrada se rellenan de acuerdo con el modo dado, en este caso se rellena con el modo reflejar a la imagen 
    )

"""
Crear Imagenes Benignas
"""
rutaB = "dataB" #Ruta de la carpeta imagenes que se va aumentar (benigno)
data_dir_list = os.listdir(rutaB)  #Lista de imagenes que se encuentra en la rutaB

b = 0          #Contador para las etiquetas diferentes de cada imagen aumentada
num_imagesB = 0 #Contador del numero imagenes total que se han generado (benigno)

for image_file in data_dir_list: #Recorremos todos las imagenes de la carpeta y tomamos el nombre de cada uno de las imagenes 
    img_list = os.listdir(rutaB)
    
    ruta_ImgB = rutaB + '/' + image_file #Concatenamos el nombre de la imagen con la direccion de la rutaB
    
    imgB = load_img(ruta_ImgB) #Cargamos la ruta de imagen a generar la transformación 
    
    imgB = cv2.resize(image.img_to_array(imgB), (128, 128), interpolation = cv2.INTER_AREA ) #Redimensionamos la imagen 
    
    x = imgB/255  
    x = np.expand_dims(x, axis = 0)
    i = 1
    
    """
    Iteramos 6 veces para las transformaciones definidas anteriormente para cada imagen benigna
    """
    for salidaB in aumento_imagenesB.flow(x, batch_size = 1):
        a = image.img_to_array(salidaB[0])
        imagen = salidaB[0,:,:]*255
        imgfinalB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cv2.imwrite(carpeta_benignoAum + "/benigno_%i%i.jpg"%(b,i), imgfinalB)
        i+=1
        
        num_imagesB+=1
        if i > num_imagsBenignoMalignoAum:
            break
    b+=1
    
print("Imagenes benignas aumentadas: ", num_imagesB)
