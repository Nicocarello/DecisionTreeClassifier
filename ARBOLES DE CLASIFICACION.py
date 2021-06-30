# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:45:23 2020

@author: Usuario
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/Anaconda/datasets/iris/iris.csv')

plt.hist(data.Species)

colnames = data.columns.values.tolist()

#AGARRO TODAS LAS COLUMNAS MENOS LA ULTIMA
predictors = colnames[:4]
#AGARRO LA ULTIMA COLUMNA QUE QUIERO PREDECIR
target = colnames[4]

import numpy as np

data['is_train']=np.random.uniform (0,1,len(data))<=0.75

train,test = data[data['is_train']==True],data[data['is_train']==False]

from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier(criterion='entropy',min_samples_split=20,random_state=99)

tree.fit(train[predictors],train[target])

preds=tree.predict(test[predictors])

#VISUALIZACION DEL ARBOL

#CREO ARCHIVO .DOT

from sklearn.tree import export_graphviz

with open('D:/Anaconda/imagenes/iris_tree.dot','w') as dotfile:
    export_graphviz(tree,out_file=dotfile,feature_names=predictors)
    dotfile.close()

#GRAFICO EL ARBOL

from sklearn.tree import plot_tree

file = open('D:/Anaconda/imagenes/iris_tree.dot','r')

text = file.read()

plot_tree(tree)


#VALIDACION CRUZADA PARA LA PODA

x= data[predictors]
y = data[target]

tree = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=20,random_state=99)

tree.fit(x,y)

print(tree.score(x, y))