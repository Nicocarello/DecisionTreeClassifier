## MACHINE LEARNING AUTOS

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('car.csv', header= None)

# LE CAMBIO LOS NOMBRES A LAS COLUMNAS DEL ARCHIVO CSV
data.columns = ['price','maintenance','puertas', 'capacidad','tamaño','seguridad','decision']

# PRINTEO LOS PRIMEROS 5 REGISTROS
print(data.head(5))
print()

# PRINTEO 5 REGISTROS AL AZAR
print(data.sample(5))
print()

# PRINTEO CANTIDAD DE FILAS Y COLUMNAS
print(data.shape)
print()

# PRINTEO LOS VALORES DE LA COLUMNA DECISION

print(data['decision'].value_counts())

# LOS ORDENO ALFABEICAMENTE

print(data['decision'].value_counts().sort_index(ascending = True))

decision = data['decision'].value_counts()
print()

# PRINTEO LOS VALORES DE COLUMNA PRICE

print(data['price'].unique())
print()

# REEMPLAZO LOS VALORES STRING A VALORES NUMERICOS

data.price.replace(('vhigh','high','med','low'),(4,3,2,1),inplace=True)
data.maintenance.replace(('vhigh','high','med','low'),(4,3,2,1),inplace=True)
data.puertas.replace(('2','3','4','5more'),(1,2,3,4),inplace=True)
data.capacidad.replace(('2','4','more'),(1,2,3),inplace = True)
data.tamaño.replace(('small','med','big'),(1,2,3),inplace = True)
data.seguridad.replace(('low','med','high'),(1,2,3),inplace = True)
data.decision.replace(('unacc','acc','good','vgood'),(1,2,3,4),inplace = True)

# PRINTEO CON LOS VALORES CAMBIADOS A NUMERICOS
print(data.head())
print()

# HAGO LA DIVISION DE LOS DATOS

# PONGO DE LA COLUMNA 0 HASTA LA 5 PARA QUE ENTRENE
dataset= data.values
X=dataset[:,1:6]
#PONGO QUE LA RTA QUE ME DEVUELVE SEA EN LA COLUMNA 6
Y=np.asarray(dataset[:,6])

# CREO EL ARBOL DE DECISION

from sklearn import tree
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

tr =tree.DecisionTreeClassifier(max_depth=10)

# ENTRENO EL MODELO
tr.fit(X_train,Y_train)

# PRUEBO SI ENTRENO EL MODELO

y_pred = tr.predict(X_test)

print(y_pred)

precision = tr.score(X_test,Y_test)

print(precision)