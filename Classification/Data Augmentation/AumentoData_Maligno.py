
#Importamos las librerias necesarias aumento de la data
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, image, img_to_array

print("Versión de tensorflow:{}".format(tf.__version__))
print("GPU:{}".format(tf.test.gpu_device_name()))

carpeta_malignoAum = 'maligno' #Crear carpeta guardar aumento imagenes maligno

num_imagsBenignoMalignoAum = 6 #Defimos el número de imagenes que se van a generar por cada imagen original (benigno Y maligno)

"""
Creamos la carpeta en el directorio en caso de no tenerla creada
"""
try:
    os.mkdir(carpeta_malignoAum)
except:
    print("")

"""
Configuramos las transformaciones que se le aplican a cada imagen para generara nuevas imagenes
"""

"""
Aumento Imagenes Malignas
"""
aumento_imagenesM = ImageDataGenerator( #Llamamos el paquete que se importo de keras, mismo que nos permite definir o configurar el generador de imagenes
    rotation_range = 30,       # Rotación: se configura en grados, se rotara la imagen a 20 grados
    zoom_range = 0.3,          #Zoom: esta configurado como una fracción, en este caso un zoom más o menos del 30% de la imagen
    width_shift_range = 0.2,   #Desplazar la imagen a lo largo de la dimensión del ancho, en este caso realizadesplazamiento del 20% de ancho de la imagen
    height_shift_range = 0.2,  #Desplazar la imagen a lo largo de la dimensión de la altura, en este caso realizadesplazamiento del 20% de ancho de la imagen
    horizontal_flip = True,   #Inversión horizontal de la imagen, es decir efecto espejo 
    vertical_flip = True,     #Inversión vertical de la imagen, es decir  efecto espejo 
    fill_mode='reflect'       #Los puntos fuera de los límites de la entrada se rellenan de acuerdo con el modo dado, en este caso se rellena con el modo reflejar a la imagen 
    )

"""
Crear Imagenes Malignas
"""
rutaM = "dataM" #Ruta de la carpeta imagenes que se va aumentar (maligno)
data_dir_list = os.listdir(rutaM) #Lista de imagenes que se encuentra en la rutaM

m = 0          #Contador para las etiquetas diferentes de cada imagen aumentada
num_imagesM = 0 #Contador del numero imagenes total que se han generado (maligno)

for image_file in data_dir_list: #Recorremos todos las imagenes de la carpeta y tomamos el nombre de cada uno de las imagenes 
    img_list = os.listdir(rutaM)
    
    ruta_ImgM = rutaM + '/' + image_file #Concatenamos el nombre de la imagen con la direccion de la rutaM
    
    imgM = load_img(ruta_ImgM) #Cargamos la ruta de imagen a generar la transformación 
    
    imgM = cv2.resize(image.img_to_array(imgM), (128, 128), interpolation = cv2.INTER_AREA ) #Redimensionamos la imagen 
    
    x = imgM/255  
    x = np.expand_dims(x, axis = 0)
    j = 1
    
    """
    Iteramos 6 veces para las transformaciones definidas anteriormente para cada imagen maligna
    """
    for salidaM in aumento_imagenesM.flow(x, batch_size = 1):
        a = image.img_to_array(salidaM[0])
        imagen = salidaM[0,:,:]*255
        imgfinalM = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cv2.imwrite(carpeta_malignoAum + "/maligno_%i%i.jpg"%(m,j), imgfinalM)
        j+=1
        
        num_imagesM+=1
        if j > num_imagsBenignoMalignoAum:
            break
    m+=1
    
print("Imagenes malignas aumentadas: ", num_imagesM)
    