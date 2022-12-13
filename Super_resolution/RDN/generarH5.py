#Enumerar imágenes y etiquetarlas
from random import shuffle as mesclar# libreria para mezclar una lista de objetos
import glob # libreria global
import numpy as np
import tables as tabla
import cv2
Mesclar_Datos = True  #  baraja las direcciones antes de guardar
Direccion_A_Guardar_En_Formato_HDF5 = 'crear/MamaPrueba.h5'  # dirección a la que desea guardar el archivo hdf5
Direccion_De_Carpeta_De_La_Data_De_Entrenamiento = 'data/data2/*.jpg' # direccion donde esta el Data de Entrenamiento

# leer direcciones y etiquetas de la carpeta 'Entrenamiento'
Direccion = glob.glob(Direccion_De_Carpeta_De_La_Data_De_Entrenamiento)
Etiquetas = [0 if 'mama' in Direc else 1 for Direc in Direccion]  # 0 = Cat, 1 = Dog

# para mezclar datos
if Mesclar_Datos:
    c = list(zip(Direccion, Etiquetas))
    mesclar(c)
    Direccion, Etiquetas = zip(*c)
    
# Divide en 60% de entrenamiento, 20% de validación y 20% de prueba
Direccion_De_Entrenamiento = Direccion[0:int(0.6*len(Direccion))]
Etiquetas_De_Entrenamiento = Etiquetas[0:int(0.6*len(Etiquetas))]

Direccion_De_Validacion = Direccion[int(0.6*len(Direccion)):int(0.8*len(Direccion))]
Etiquetas_De_Validacion = Etiquetas[int(0.6*len(Direccion)):int(0.8*len(Direccion))]

Direccion_De_Prueba = Direccion[int(0.8*len(Direccion)):]
Etiquetas_De_Prueba = Etiquetas[int(0.8*len(Etiquetas)):]
#Archivo_HDF5.close()

print("Proceso de enumerado y etiquetado imagenes con exito!") #Imprime mensasje







#Creando un archivo HDF5

Orden_De_Datos = 'tf'  # 'th' para Theano, 'tf' para Tensorflow 

Tipo_De_Imagen = tabla.UInt8Atom() # dtype en el que se guardarán las imágenes  

# verifica el orden de los datos y elige la forma de datos adecuada para guardar las imágenes
if Orden_De_Datos == 'th':
    Forma_De_Dato = (0, 3, 224, 224)
elif Orden_De_Datos == 'tf':
    Forma_De_Dato = (0, 224, 224, 3)

# abrir un archivo hdf5 y crear auriculares
Archivo_HDF5 = tabla.open_file(Direccion_A_Guardar_En_Formato_HDF5, mode='w')

Almacenamiento_De_Entrenamiento = Archivo_HDF5.create_earray(Archivo_HDF5.root, 'Imagen_De_Entrenamiento', Tipo_De_Imagen, shape=Forma_De_Dato)
Almacenamiento_De_Validacion = Archivo_HDF5.create_earray(Archivo_HDF5.root, 'Imagen_De_Validacion', Tipo_De_Imagen, shape=Forma_De_Dato)
Almacenamiento_De_Prueba = Archivo_HDF5.create_earray(Archivo_HDF5.root, 'Imagen_De_Prueba', Tipo_De_Imagen, shape=Forma_De_Dato)

Almacenamiento_Medio = Archivo_HDF5.create_earray(Archivo_HDF5.root, 'Medio_Entrenamiento', Tipo_De_Imagen, shape=Forma_De_Dato)

# crear las matrices de etiquetas y copiar los datos de las etiquetas en ellas
Archivo_HDF5.create_array(Archivo_HDF5.root, 'Etiquetas_De_Entrenamiento', Etiquetas_De_Entrenamiento)
Archivo_HDF5.create_array(Archivo_HDF5.root, 'Etiquetas_De_Validacion', Etiquetas_De_Validacion)
Archivo_HDF5.create_array(Archivo_HDF5.root, 'Etiquetas_De_Prueba', Etiquetas_De_Prueba)

print("Archivo HDF5 Creado con exito") #Imprime mensasje








# una matriz numpy para guardar la media de las imágenes
Media = np.zeros(Forma_De_Dato[1:], np.float32)

# loop over train addresses
for i in range(len(Direccion_De_Entrenamiento)):
    # imprimir cuántas imágenes se guardan cada 1000 imágenes
    if i % 10 == 0 and i > 1:
        #print ('Datos de Entrenamiento: {}/{}'.format(i, len(Direccion_De_Entrenamiento)))
        print ("Datos de Entrenamiento: {}/{}".format(i, len(Direccion_De_Entrenamiento)))

    # leer una imagen y cambiar el tamaño a (224, 224)
    # cv2 carga imágenes como BGR, conviértala a RGB
    try:
        Direc = Direccion_De_Entrenamiento[i]
        Imagen = cv2.imread(Direc)
        Imagen = cv2.resize(Imagen, (224, 224), interpolation=cv2.INTER_CUBIC)
        Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2RGB)
        print("el shape de la imagen es   ",Imagen.shape)
        if Orden_De_Datos == 'th':
            Imagen = np.rollaxis(Imagen, 2)

    # guardar la imagen y calcular la media hasta ahora
            Almacenamiento_De_Entrenamiento.append(Imagen[None])
            Media += Imagen / float(len(Etiquetas_De_Entrenamiento))
    except Exception as e:
        print(str(e))

    
      # agregar cualquier preprocesamiento de imagen aquí
    
    # si el orden de los datos es Theano, los pedidos del eje deberían cambiar
    

# loop sobre las direcciones de validación
for i in range(len(Direccion_De_Validacion)):
    # imprimir cuántas imágenes se guardan cada 1000 imágenes
    if i % 10 == 0 and i > 1:
        print ('Datos de Validacion: {}/{}'.format(i, len(Direccion_De_Validacion)))

    # leer una imagen y cambiar el tamaño a (224, 224)
    # cv2 carga imágenes como BGR, conviértala a RGB
    try:
        Direc = Direccion_De_Validacion[i]
        Imagen = cv2.imread(Direc)
        Imagen = cv2.resize(Imagen, (224, 224), interpolation=cv2.INTER_CUBIC)
        Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2RGB)
        print("el shape de la imagen es   ",Imagen.shape)
    # agregar cualquier preprocesamiento de imagen aquí
    
    except Exception as e:
        print(str(e))

    # si el orden de los datos es Theano, los pedidos del eje deberían cambiar
    if Orden_De_Datos == 'th':
        Imagen = np.rollaxis(Imagen, 2)

    # guardar la imagen
    Almacenamiento_De_Validacion.append(Imagen[None])

# loop over test addresses
for i in range(len(Direccion_De_Prueba)):
    # imprimir cuántas imágenes se guardan cada 1000 imágenes
    if i % 10 == 0 and i > 1:
        print ('Datos de Prueba: {}/{}'.format(i, len(Direccion_De_Prueba)))

    # leer una imagen y cambiar el tamaño a (224, 224)
    # cv2 carga imágenes como BGR, conviértala a RGB
    try: 
        Direc = Direccion_De_Prueba[i]
        Imagen = cv2.imread(Direc)
        Imagen = cv2.resize(Imagen, (224, 224), interpolation=cv2.INTER_CUBIC)
        Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2RGB)
        print("el shape de la imagen es   ",Imagen.shape)
        if Orden_De_Datos == 'th':
            Imagen = np.rollaxis(Imagen, 2)

            # guardar la imagen
            Almacenamiento_De_Prueba.append(Imagen[None])
    except Exception as e:
        print(str(e))

    # agregar cualquier preprocesamiento de imagen aquí
    
    # si el orden de los datos es Theano, los pedidos del eje deberían cambiar
   

# guardar la media y cerrar el archivo hdf5
Almacenamiento_Medio.append(Media[None])
Archivo_HDF5.close()

print("Imagenes Guardadas con exito en Formato HDF5") #Imprime mensasje

print ('Total de Imagenes: {}'.format(len(Direccion_De_Prueba + Direccion_De_Entrenamiento+ Direccion_De_Validacion)))