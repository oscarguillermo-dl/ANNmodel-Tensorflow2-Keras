"""ModeloRNA.ipynb

#Importar Librerías
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import metrics
from sklearn.metrics import r2_score
import pandas as pd

from google.colab import files

"""#Carga de los sets de entrenamiento y prueba"""

uploaded = files.upload() #Cargar los archivos que contienen los sets de entrenamiento y prueba

"""#Carga de los datos de los sets de entrenamiento y prueba en variables"""

enviromentaldata_train = 'enviromentaldata_train.xlsx' #Cargar archivo de datos de entrada del set de entrenamiento
dfenvdata_train = pd.read_excel(enviromentaldata_train) #Leer los datos en formato excel
envdata_train = dfenvdata_train.values #Guardar los datos en una variable
performancedata_train = 'performancedata_train.xlsx' #Cargar archivo de datos de salida del set de entrenamiento
dfperfdata_train = pd.read_excel(performancedata_train) #Leer los datos en formato excel
perfdata_train = dfperfdata_train.values #Guardar los datos en una variable

enviromentaldata_test = 'enviromentaldata_test.xlsx' #Cargar archivo de datos de entrada del set de prueba
dfenvdata_test = pd.read_excel(enviromentaldata_test) #Leer los datos en formato excel
envdata_test = dfenvdata_test.values #Guardar los datos en una variable
performancedata_test = 'performancedata_test.xlsx' #Cargar archivo de datos de salida del set de prueba
dfperfdata_test = pd.read_excel(performancedata_test)#Leer los datos en formato excel
perfdata_test = dfperfdata_test.values #Guardar los datos en una variable

"""#Normalización de los datos de entrada, creación del modelo y evaluación con coficiente de determinación"""

normalizer = layers.Normalization() #Creación de función para normalizar
normalizer.adapt(envdata_train) #Adaptación de los datos a los valores de entrada del set de entrenamiento

model = keras.Sequential( #Creación del modelo con Sequential
    [
        normalizer, #Primera capa dpara normalizar los datos de entrada
     layers.Dense(10,activation='relu',name='CapaOculta1'), #Primera capa oculta (No neuronas, función de activación, nombre)
     layers.Dense(10,activation='relu',name='CapaOculta2'), #Segunda capa oculta (No neuronas, función de activación, nombre)
     layers.Dense(10,activation='relu',name='CapaOculta3'), #Tercera capa oculta (No neuronas, función de activación, nombre)
     layers.Dense(1,name='Salida') #Capa de salida (No neuronas, nombre)
    ],name = 'ModeloRNA' #Nombre de la red neronal
)

model.compile(optimizer='adam', #Compilación del modelo (Optimizador)
loss = 'mse', #Evaluador de pérdida
metrics = [metrics.mean_squared_error]) #Métrica para evalaución de ajuste del modelo (Adiconal al coeficiente de determinación)

model.fit(envdata_train,perfdata_train,epochs=3) #Entrenamiento del modelo con las propiedades asignadas anteriormente

perfdata_pred = model.predict(envdata_train) #Prueba del modelo con el set de entrenamiento para evaluación

r2 = r2_score(perfdata_train,perfdata_pred) #Cálculo del coeficiente de determinación relacionando los valores reales (perfdata_train) y los estimados(perfdata_pred)
print('R2:',r2)

"""#Resumen del modelo creado (Capas, Parámetros Básicos)"""

model.summary() #Resumen del modelo creado

"""#Descarga de los parámetros del modelo para implementar en Arduino"""

model.save_weights('model_rna.h5') #Guardar en un archivo .h5 los parámetros de la red (pesos entre conexiones, sesgos de cada neurona)
