
import numpy as np
import cv2
import sys

def obtener_img(ruta, ancho, altura, ordering):
    try:
        img = cv2.imread(ruta, 1) #1  para rgb - 0 para gris
        img = cv2.resize(img, ( ancho , altura ))
        img = img/255.0
    except Exception:
        print("Error inesperado:", sys.exc_info()[0])
        img = np.zeros(( altura , ancho , 3 ))
        
    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
        
    return img

def obtener_seg(ruta, ancho, altura, ordering):
    etiqueta_seg = np.zeros((altura, ancho, 1))
    
    try:
        img = cv2.imread(ruta, 0) # 0 para gris
        img = cv2.resize(img, ( ancho , altura ))
        img = img/255.0
        etiqueta_seg[:, :,0] = (img >0.5 ).astype(int)
        
    except Exception:

        print("Error inesperado:", sys.exc_info()[0])
    
    if ordering == 'channels_first':
        etiqueta_seg = np.rollaxis(img, 2, 0)
        
    return etiqueta_seg