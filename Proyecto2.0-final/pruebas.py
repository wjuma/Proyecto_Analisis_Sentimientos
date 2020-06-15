import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd


# CARGAR DATASET DE DROPBOX
#-----------------------------------------------------------------
data = pd.read_csv('https://www.dropbox.com/s/cugxdc9mhau4nw1/titanic2.csv?dl=1')
#data = data.as_matrix()
data = np.matrix(data)


# CREA dataset TRAIN y TEST
#---------------------------------------------------------------------------------------------
np.random.seed(123)
m_train    = np.random.rand(len(data)) < 0.5
data_train = data[m_train,]
data_test  = data[~m_train,]


# CLASE
#---------------------------------------------------------------------------------------------
clase_train = data_train[:,-1]
clase_train = clase_train.A1 #convierte de matriz a vector
clase_test  = data_test[:,-1]
clase_test  = clase_test.A1 #convierte de matriz a vector


# MODELO
#---------------------------------------------------------------------------------------------
modelo_lr = LogisticRegression()
modelo_lr.fit(X=data_train[:,:-1],y=clase_train)


# PREDICCION
#---------------------------------------------------------------------------------------------
predicion = modelo_lr.predict(data_test[:,:-1])


# METRICAS
#---------------------------------------------------------------------------------------------
print(metrics.classification_report(y_true=clase_test, y_pred=predicion))
print(pd.crosstab(data_test[:,-1].A1, predicion, rownames=['REAL'], colnames=['PREDICCION']))