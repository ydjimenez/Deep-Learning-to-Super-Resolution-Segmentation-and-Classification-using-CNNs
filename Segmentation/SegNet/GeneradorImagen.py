
from LecturaImagen import obtener_img, obtener_seg
import numpy as np
import itertools
import glob

def generador_img( ruta_img, ruta_mascara, tamanio_lote, altura, ancho, ordering):
    try:
        assert ruta_img[-1] == '/'
        assert ruta_mascara[-1] == '/'
        
        images = glob.glob(ruta_img + "Mamo_*.jpg")
        images.sort()
        
        segmentations = glob.glob(ruta_mascara + "Mamo_*.png")
        segmentations.sort()
        
        assert len(images) == len(segmentations) 
        
        zipped = itertools.cycle( zip(images,segmentations) )
        
        while True:
            X = []
            Y = []
            for _ in range(tamanio_lote):
                im , seg = next(zipped)
                X.append(obtener_img(im, ancho, altura, ordering))
                Y.append(obtener_seg(seg, ancho, altura,ordering))
                
            yield np.array(X) , np.array(Y)
    except StopIteration:
        pass
        