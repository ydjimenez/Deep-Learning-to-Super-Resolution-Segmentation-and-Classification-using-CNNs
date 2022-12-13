from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools

print("Versión de tensorflow:{}".format(tf.__version__))
print("GPU:{}".format(tf.test.gpu_device_name()))

ancho, altura = 128, 128
batch_size = 1

names = ['benigno','maligno']

test_data_dir = './data/pruebas'  

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(ancho, altura), 
    batch_size = batch_size,
    class_mode='categorical', 
    shuffle=False)

custom_Model= load_model("./model/modelResnet50_breast.h5")

predictions = custom_Model.predict_generator(generator=test_generator)

y_pred = np.argmax(predictions, axis=1)
y_real = test_generator.classes

"""
MATRIZ DE CONFUSIÓN - EVALUAR LA CNN
"""

cm = confusion_matrix(y_real, y_pred) # La matriz de ocnfusion le pasa como parametros los indices reales de las salidas de las imagenes 
                                                                   # y los resultados predichos de nuestra red neurona convolucional 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y traza la matriz de confusión.
    La normalización se puede aplicar configurando `normalize = True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
cm_plot_labels = test_generator.class_indices
plot_confusion_matrix(cm, cm_plot_labels, title = "Matriz de confusión")

"""
PARA COMPARAR LAS PREDICCIONES DE LAS IMAGENES
"""

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in y_pred] 

"""
Se genera un exel para comprar, donde, se encuentran el nombre del archivo y la prediccion 
"""
filenames = test_generator.filenames
results = pd.DataFrame({"Ruta archivo":filenames,
                        "Prediccion":predictions})
results.to_csv("archivos/Resultados_clasificación.csv", index = False)
 
"""
Métricas de evaluación
"""
print("Métricas de evaluación")
Etiqueta = ['Benigno 0', 'Maligno 1']
metricas = metrics.classification_report(y_real,y_pred, digits = 2, target_names=Etiqueta)
print(metricas)

"""
Extraccion: TN, FP, FN, TP
"""
tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()

#Exactitud
Exactitud = (tn+tp)*100/(tp+tn+fp+fn) 
print("Exactitud: {:0.2f}%".format(Exactitud))

#Precision 
Precision = tp/(tp+fp) 
print("Precision: {:0.2f}".format(Precision))

#Recall ó Sensibilidad
Sensibilidad = tp/(tp+fn) 
print("Sensibilidad: {:0.2f}".format(Sensibilidad))

#Specificity 
Especificidad = tn/(tn+fp)
print("Especificidad: {:0.2f}".format(Especificidad))

#F1 Score
F1 = (2*Precision*Sensibilidad)/(Precision + Sensibilidad)
print("F1 Score: {:0.2f}".format(F1))







 