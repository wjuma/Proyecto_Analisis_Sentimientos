import punto2 as p
from sklearn.linear_model import LogisticRegression

X = p.x
Y = p.y
data=p.data

Xtraining=X[:700]
Xtest=X[700:]
Ytraining=Y[:700]
Ytest=Y[700:]





# Regresión logistica training

model = LogisticRegression(solver='liblinear', random_state=0).fit(Xtraining, Ytraining)

modelo_lr = LogisticRegression(solver='lbfgs')
modelo_lr.fit(Xtraining,Ytraining)

#y_predict_lr = modelo_lr.predict(Ytest)

#print(accuracy_score(y111, y_predict_lr))

#modelo_lr = LogisticRegression()
#Regresion logistica 
#modelo_lr.fit(Xtraining,Ytraining)

# print('Independent term: \n', modelo_lr.intercept_)
# print('Coefficients: \n', modelo_lr.coef_)
mode = "";
for i in range(len(modelo_lr.coef_[0])):
    mode +=' + ({} * X{})'.format(round(modelo_lr.coef_[0][i],3),i+1)


print('********************************************************************')
print('***************************   Y-ESTIMADA     ***********************')
print('********************************************************************')

print('Y =','e','^',modelo_lr.intercept_, mode,'/','1  + ','e','^',modelo_lr.intercept_, mode)

print('____________________________________________________________________')


print('********************************************************************')
print('****************************  Predicción    ************************')
print('********************************************************************')
y_predict_lr1=modelo_lr.predict(Xtest)

print()
print(y_predict_lr1)

#print("-------->",len(y_predict_lr1))
print('____________________________________________________________________')
print('____________________________________________________________________')


print('********************************************************************')
print('***************************    Predicción     ***********************')
print('********************************************************************')


for i in range(len(p.y[700:].index)):
    ind=p.y[700:].index[i]
    
    print('{}{} {}'.format("Twitter de covid(Training)",ind,':'))
    print(data['text'][ind])
    print('')
    print('{} {}: --->{}'.format("Predicción-Y ",ind,y_predict_lr1[i]))
    print('')
    print('{} {}: --->{}'.format("Dataset-Y ",ind,data['target'][ind]))
    print('\n')

'''
    if y_predict_lr1[i]==1:
        print(data['text'][ind],'Positivo')
        
    else:
        print(data['text'][ind], 'Negativo')
'''

print('********************************************************************')
print('**************************  Precisión del modelo  ******************')##
print('********************************************************************')
#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(Ytest, y_predict_lr1)
print('Precisión del modelo:')
print(precision)

print('********************************************************************')
print('**************************  Exactitud del modelo  ******************')
print('********************************************************************')
#Calculo la exactitud del modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(Ytest,  y_predict_lr1)
print('Exactitud del modelo:')
print(exactitud)

print('********************************************************************')##
print('*****************************   Error  *****************************')
print('********************************************************************')
# Calculando error

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Ytest, y_predict_lr1)
print('Error cuadratico medio :')
print(mse)

