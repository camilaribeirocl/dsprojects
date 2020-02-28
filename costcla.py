# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:46:57 2020

@author: camila.ribeiro
"""
import pyodbc
import pandas as pd
import numpy as np
#lendo o banco de dados
cnxn = pyodbc.connect(r'Driver={SQL Server};Server=brtvoipbio;DATABASE=brt_aux;Trusted_Connection=yes')
query = "select * from [brt_aux].[dbo].brt_aux_base_cartao_internet_treino_teste_dummy"
dataset = pd.read_sql(query, cnxn)
cnxn.close()
#removendo colunas com muitos na's 
dataset2 = dataset.drop(['fraud_score', 'idade_dominio',
                  'idade_email'], axis=1)
dataset2 = dataset2.dropna(axis = 0, how = 'any')
dataset2.reset_index(drop=True, inplace=True)
dataset2.columns
#remocao de features com muitos na's
#X = dataset2.drop(['transacao_id','transacao_fraude'], axis=1)
X = dataset2[['pessoa_idade','pessoa_sexo_f','pessoa_sexo_m','transacao_valor',
              'score_konduto']]
y = dataset2['transacao_fraude']
list(X.columns)
cols = X.columns #.drop('id')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
# cost_mat[C_FP,C_FN,C_TP,C_TN]
#criando data frame
cost_mat = pd.DataFrame(np.zeros((52036, 4)))
cost_mat[0] = X['transacao_valor']
cost_mat[1] = 2
cost_mat[2] = 2
cost_mat[3] = 2

cost_mat = X['transacao_valor']
cost_mat['C_FN'] = X['transacao_valor']
cost_mat['C_FP'] = 0.0
cost_mat['C_TP'] = 0.0
cost_mat['C_TN'] = 0.0

X= X.to_numpy()
y= y.to_numpy()
cost_mat = cost_mat.to_numpy()
#fit the classifiers 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = train_test_split(X, y, cost_mat)

# Fit the classifiers using the training dataset
classifiers = {"RF": {"f": RandomForestClassifier()},
               "DT": {"f": DecisionTreeClassifier()},
               "LR": {"f": LogisticRegression()}}

for model in classifiers.keys():
    # Fit
    classifiers[model]["f"].fit(X_train, y_train)
    # Predict
    classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
    classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
    classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(X_train)

# Evaluate the performance
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
measures = {"f1": f1_score, "pre": precision_score, 
            "rec": recall_score, "acc": accuracy_score}
results = pd.DataFrame(columns=measures.keys())



from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score

y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
f = CostSensitiveDecisionTreeClassifier()
y_pred_test_csdt = f.fit(X_train,y_train,cost_mat_train)
# Savings using only RandomForest
print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
X_train.values
y_train.to_numpy()


print(savings_score(y_test, y_pred_test_rf, cost_mat_test))












