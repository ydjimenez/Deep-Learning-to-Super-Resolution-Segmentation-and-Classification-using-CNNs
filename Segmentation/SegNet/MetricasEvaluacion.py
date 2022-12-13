
import numpy as np
import math

def metricas_evaluacion(masca, predic):
    mascara = masca[:,:,0]
    prediccion = predic[:,:,0]

    npixels = mascara.size

    tp = np.sum(mascara*prediccion)

    fp = np.sum((prediccion-mascara)==1)

    fn = np.sum((mascara-prediccion)==1)

    tn = npixels-tp-fp-fn
    
    exactitud  = (tp+tn)/(tp+fp+tn+fn)

    if math.isnan(exactitud ):
        exactitud =0.0
    
    precision = tp/(tp+fp)

    if math.isnan(precision):
        precision=0.0
        
    sensibilidad = tp/(tp+fn)

    if math.isnan(sensibilidad):
        sensibilidad=0.0
        
    especificidad   = tn/(tn+fp)

    if math.isnan(especificidad  ):
        especificidad  =0.0
        
    dice = 2*tp/(2*tp+fn+fp)

    if math.isnan(dice):
        dice=0.0
        
    jaccard = tp/(tp+fn+fp)

    if math.isnan(jaccard ):
        jaccard=0.0
        
    metricas = [exactitud , precision, sensibilidad, especificidad, dice, jaccard]
    
    return metricas