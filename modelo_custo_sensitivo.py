#importando bibliotecas
import pandas as pd
import numpy as np

#lendo o banco de dados
dataset = pd.read_csv('C:/Users/Camila Ribeiro/Documents/kaggle/creditcardfinal.csv')

#contando na's por coluna 
dataset.isnull().sum(axis = 0)
#remocao de features

X = dataset.drop(['Class','Time','V1'], axis=1)
y = dataset['Class']
print (X.dtypes)

# cost_mat[C_FP,C_FN,C_TP,C_TN]
# criando matriz de custos 
cost_mat = pd.DataFrame(np.zeros((284807, 4)))
cost_mat[0] = X['Amount']
cost_mat[1] = 300
cost_mat[2] = 300
cost_mat[3] = 300

#to numpy 
X= X.to_numpy()
y= y.to_numpy()
cost_mat = cost_mat.to_numpy()

#treino e teste
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = train_test_split(X, y, cost_mat)

from costcla.models import CostSensitiveRandomForestClassifier
from costcla.metrics import savings_score

#testando random forest e random forest 
y_pred_test_rf = RandomForestClassifier(class_weight= 'balanced',random_state=0).fit(X_train, y_train).predict(X_test)
f = CostSensitiveRandomForestClassifier(n_jobs =-1)
y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
# Savings using only RandomForest
print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
# Savings using CostSensitiveRandomForestClassifier
print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))

#testando regressao logistica
from costcla.models import CostSensitiveLogisticRegression
f = CostSensitiveLogisticRegression(C= 1000000, max_iter= 1000,fit_intercept =False)
f.fit(X_train, y_train, cost_mat_train)
y_pred_test_cslr = f.predict(X_test)

print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
# Savings using CostSensitiveRandomForestClassifier
print(savings_score(y_test, y_pred_test_cslr, cost_mat_test))

from costcla.datasets import load_creditscoring1
data = load_creditscoring1()

#metricas 
from sklearn.metrics import accuracy_score
threshold = 0.5
predicted_proba = f.predict_proba(X_test)
predicted = (predicted_proba [:,1] >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
accuracy

#metricas 
import sklearn.metrics as m
print(m.classification_report(y_test,predicted))
print(m.confusion_matrix(y_test, predicted))
