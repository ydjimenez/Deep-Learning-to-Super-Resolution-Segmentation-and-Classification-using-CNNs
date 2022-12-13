
from keras.models import load_model
from LecturaImagen import obtener_img, obtener_seg
from MetricasEvaluacion import metricas_evaluacion
import pandas as pd
import numpy as np
import glob
import cv2

DIM_ORDER='channels_last'
guardar_prediccion = True
guardar_metrica = True

ruta_img_test = "Datos/Validacion/RoiImg/"
ruta_mascara_test = "Datos/Validacion/RoiMascara/"
tamanio_test = 156 

ruta_modelo_entrenado = "ModeloEntrenado/modelo_entrenado.h5" 
ancho= 128
altura = 128

modelo = load_model(ruta_modelo_entrenado)

imgs = glob.glob( ruta_img_test + "*.jpg" )
imgs.sort() 

mascara = glob.glob( ruta_mascara_test + "*.png" )
mascara.sort() 

titulo = ['Accuracy', 'Precision', 'Sensitivity o recall', 'Specificity', 'Sorensen-Dice coefficient', 'Jaccard index']

index = []
list_metricas = []

i=0
indice=0

for imagen in imgs:
    # Predecir segmentación
    X = obtener_img(imagen, ancho, altura, DIM_ORDER)
    
    predic = modelo.predict( np.array([X]), verbose=0)[0]
    predic = (predic>0.5).astype(int)   
    
    seg_img = imagen.replace(ruta_img_test,ruta_mascara_test).replace(".jpg", "_seg.png")
    
    while ((indice < tamanio_test) and (i == indice)):
        masca = obtener_seg(mascara[indice], ancho, altura, DIM_ORDER)
        indice = indice + 1
        
    metrics = metricas_evaluacion(masca, predic)
    list_metricas.append(metrics)

    if guardar_prediccion:
        index.append(imagen.replace(ruta_img_test, ""))
        predic_img = predic*255
        cv2.imwrite(seg_img, predic_img)
    
    i=i+1
    
metricas_df = pd.DataFrame(list_metricas, index=index, columns=titulo)
metricas = pd.DataFrame(list_metricas, index=index, columns=titulo)
metrica_total = metricas_df.mean(axis=0, numeric_only=True)
print('\nMétricas de evaluación:')
print(metrica_total)

if guardar_metrica:
    metricas_df.loc['Media'] = metrica_total
    metricas_df.to_csv("Archivos/Métricas_evaluación.csv", mode="w+", index=True, header=True)    
