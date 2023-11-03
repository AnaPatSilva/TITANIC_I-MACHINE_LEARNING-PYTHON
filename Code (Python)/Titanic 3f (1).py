# -*- coding: utf-8 -*-
"""
Vou dividir o dataset de treino (em treino e teste) para conseguir avaliar o modelo
Vou continuar com as 3 features porque tiveram uma melhor performance
"""

#%% Instalação de livrarias
#%pip install pandas
##%pip install pandas_profiling #inativo
#%pip install ydata_profiling
#%pip install scikit-learn


#%% Importação de datasets

import os
import pandas as pd

for dirname, _, filenames in os.walk('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

gender = pd.read_csv('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data\gender_submission.csv')
test = pd.read_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Data/test.csv') # tive de mudar a direção da barra, porque estava a dar erro
train = pd.read_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Data/train.csv')

print(gender.head())
print(test.head())
print(train.head())

print(train.shape)
train.describe()

#%% Data Profiling do dataset de treino (comentado para não gerar relatório sempre que corro o código)

#from ydata_profiling import ProfileReport

#profile_train = ProfileReport(train, title="Profile Train")
#profile_train.to_file("profile_train.html")

#Há uma correlação grande entre Survived e Sex

#Ver as linhas com valores nulos
NA = train[train.isna().any(axis=1)]
print(NA)
#só aparecem missing values na coluna da Cabin

# Vou usar as variáveis Sex, Pclass, Fare
train['Pclass'].value_counts() #Verificar a distribuição da feature no dataset
train['Pclass'].value_counts(normalize=True)
train['Fare'].value_counts()
train['Fare'].value_counts(normalize=True)

NA_Pclass = train['Pclass'].isnull()
NA_Pclass_sum = NA_Pclass.sum()
print(f'Número de valores nulos em "Pclass": {NA_Pclass_sum}')

NA_Fare = train['Fare'].isnull()
NA_Fare_sum = NA_Fare.sum()
print(f'Número de valores nulos em "Fare": {NA_Fare_sum}')

#Nesta fase, não vou utilizar o dataset "test"

#NA_Sex = test['Sex'].isnull()
#NA_Sex_sum = NA_Sex.sum()
#print(f'Número de valores nulos em "Sex": {NA_Sex_sum}')

#NA_Fare_test = test['Fare'].isnull()
#NA_Fare_test_sum = NA_Fare_test.sum()
#print(f'Número de valores nulos em "Fare": {NA_Fare_test_sum}')

#NA_Fare_test_list = test.loc[test['Fare'].isnull()]
#print(NA_Fare_test_list)

# eliminar a linha com valor nulo e criar um novo dataset
#test_na = test.dropna(subset=['Fare'])


#%%Divisão dos dados e definição da Feature (Sex, Pclass, Fare) e do Target/Label (Survived)

from sklearn.model_selection import train_test_split

#y = target
y_target = train['Survived']
#x = features
x_features = pd.get_dummies(train[['Sex', 'Pclass', 'Fare']]) # realiza uma codificação one-hot num DataFrame, usando um conjunto de características (features). A codificação one-hot é uma técnica usada para transformar variáveis categóricas em representações numéricas binárias (0, 1).

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.3, random_state=42) #dividiu em 70% treino e 30% teste


#%% Treinamento do modelo inicial de Random Forest

from sklearn.ensemble import RandomForestClassifier

#n_estimators (nº de árvores) = 100
#max_depth (quantidade de camadadas) = 5
#random_state (seed) = 42
RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
RF.fit(x_train, y_train)
y_prediction_target = RF.predict(x_test)


#%% Medição da precisão do modelo inicial

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_prediction_target)
print(f'Precisão do modelo: {accuracy:.2f}')


#%% Cross Validation para evitar o over fitting

from sklearn.model_selection import cross_val_score, KFold

# número de dobras = 5
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
scores_kf = cross_val_score(RF, x_train, y_train, cv=kf, scoring='accuracy')
print("Precisão para cada fold:", ["%.2f" % score for score in scores_kf])
print("Precisão média: %.2f" % scores_kf.mean())


from sklearn.model_selection import cross_val_score, RepeatedKFold

# número de repetições = 3
num_repeats = 3
rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
scores_rkf = cross_val_score(RF, x_train, y_train, cv=rkf, scoring='accuracy')
print("Precisão para cada fold:", ["%.2f" % score for score in scores_rkf])
print("Precisão média: %.2f" % scores_rkf.mean())


#%% Tuning do modelo inicial de Random Forest

# Definir a grade de hiperparâmetros que você deseja pesquisar
param_grid = {
    'n_estimators': [100, 200, 300], #nº de árvores
    'max_depth': [None, 10, 20, 30], #nº de camadas
    'min_samples_split': [2, 5, 10], #nº mínimo de amostras necessárias para dividir um nó interno da árvore
    'min_samples_leaf': [1, 2, 4] #nº mínimo de amostras necessárias para que uma folha (nó terminal) seja criada
}

# Primeiro usar o Random Search para pesquisar os hiperparâmetros

from sklearn.model_selection import RandomizedSearchCV

#cv = cross validation a usar (estou a usar a rkf que fiz anteriormente)
random_search = RandomizedSearchCV(estimator=RF, param_distributions=param_grid, n_iter=10, cv=rkf, scoring='accuracy', n_jobs=-1, random_state=42)

# Realizar a pesquisa aleatória
random_search.fit(x_train, y_train)

# Obter os melhores hiperparâmetros encontrados
melhores_hiperparametros_rs = random_search.best_params_

# Obter o melhor estimador (modelo)
melhor_modelo_rs = random_search.best_estimator_

# Obtenha a melhor pontuação (score)
melhor_pontuacao_rs = random_search.best_score_

print("Melhores Hiperparâmetros:", melhores_hiperparametros_rs)
print("Melhor Estimador (Modelo):", melhor_modelo_rs)
print("Melhor Pontuação (Score): %.2f" % melhor_pontuacao_rs)


# Agora vou usar o GridSearchCV para pesquisar os hiperparâmetros e comparar com o Random Search

from sklearn.model_selection import GridSearchCV

#cv = cross validation a usar (estou a usar a rkf que fiz anteriormente)
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=rkf, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)
melhores_hiperparametros_gs = grid_search.best_params_

# Obter o melhor modelo encontrado pelo GridSearch
melhor_modelo_gs = grid_search.best_estimator_

# Avaliar o melhor modelo nos dados de teste
acuracia_teste = melhor_modelo_gs.score(x_test, y_test)

# Obtenha a melhor pontuação (score)
melhor_pontuacao_gs = grid_search.best_score_

# Exibir os melhores hiperparâmetros e a acurácia no conjunto de teste
print("Melhores Hiperparâmetros:", melhores_hiperparametros_gs)
print("Melhor Estimador (Modelo):", melhor_modelo_gs.get_params())
print("Melhor Pontuação (Score): %.2f" % melhor_pontuacao_gs)
print("Acurácia no Conjunto de Teste: %.2f" % acuracia_teste)


#%% Treinamento do modelo de Random Forest ajustado (tuned)
#Como ambos os tuning tiveram o mesmo resultado vou treinar com ambos os hiperparametros

RF_RS = RandomForestClassifier(
    n_estimators=melhores_hiperparametros_rs['n_estimators'],
    max_depth=melhores_hiperparametros_rs['max_depth'],
    min_samples_split=melhores_hiperparametros_rs['min_samples_split'],
    min_samples_leaf=melhores_hiperparametros_rs['min_samples_leaf'],
    random_state=42  # Defina a semente aleatória se necessário
)
RF_RS.fit(x_train, y_train)
y_prediction_target_rs = RF_RS.predict(x_test)


RF_GS = RandomForestClassifier(
    n_estimators=melhores_hiperparametros_gs['n_estimators'],
    max_depth=melhores_hiperparametros_gs['max_depth'],
    min_samples_split=melhores_hiperparametros_gs['min_samples_split'],
    min_samples_leaf=melhores_hiperparametros_gs['min_samples_leaf'],
    random_state=42  # Defina a semente aleatória se necessário
)
RF_GS.fit(x_train, y_train)
y_prediction_target_gs = RF_GS.predict(x_test)


#%% Medição da precisão do modelo após o ajuste

accuracy_RS = accuracy_score(y_test, y_prediction_target_rs)
print(f'Precisão do modelo (Random Search): {accuracy_RS:.2f}')

accuracy_GS = accuracy_score(y_test, y_prediction_target_gs)
print(f'Precisão do modelo (Grid Search): {accuracy_GS:.2f}')


# Vou fazer uma curva ROC do Grid Search

from sklearn.metrics import roc_curve, roc_auc_score

y_probs = RF_GS.predict_proba(x_test)

# Calcule as taxas de verdadeiro positivo (TPR) e as taxas de falso positivo (FPR) usando as probabilidades previstas e os rótulos reais
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])

# Calcule a área sob a curva ROC (AUC-ROC)
auc_roc = roc_auc_score(y_test, y_probs[:, 1])

# Plote a curva ROC

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Exiba o valor da AUC-ROC
print("AUC-ROC: %.2f" % auc_roc)


#%% Matriz de Confusão

from sklearn.metrics import confusion_matrix

# 'y_test' sejam as classes reais e 'y_pred' sejam as previsões do modelo
conf_matrix = confusion_matrix(y_test, y_prediction_target_gs)

print("Matriz de Confusão:")
print(conf_matrix)


#%% Resumo dos resultados
print(f'Precisão do modelo inicial: {accuracy:.2f}') #0,79
print("Precisão média Cross Folding: %.2f" % scores_kf.mean()) #0,81
print("Precisão média Repetated Cross Folding: %.2f" % scores_rkf.mean()) #0,81
print("Melhor Pontuação (Score) Random Search: %.2f" % melhor_pontuacao_rs) #0,81
print("Melhor Pontuação (Score) Grid Search: %.2f" % melhor_pontuacao_gs) #0,81
print(f'Precisão do modelo (Random Search): {accuracy_RS:.2f}') #0,82
print(f'Precisão do modelo (Grid Search): {accuracy_GS:.2f}') #0,82
print("AUC-ROC: %.2f" % auc_roc) #0,84


#%% Aplicar o melhor modelo ao dataset "Test"

NA_Sex = test['Sex'].isnull()
NA_Sex_sum = NA_Sex.sum()
print(f'Número de valores nulos em "Sex": {NA_Sex_sum}')

NA_Fare_test = test['Fare'].isnull()
NA_Fare_test_sum = NA_Fare_test.sum()
print(f'Número de valores nulos em "Fare": {NA_Fare_test_sum}')

NA_Fare_test_list = test.loc[test['Fare'].isnull()]
print(NA_Fare_test_list)

# eliminar a linha com valor nulo e criar um novo dataset
#test_na = test.dropna(subset=['Fare'])


# Vou substituir o valor nulo em falta no Fare, porque o output do dataset tem de ter 418 linhas
# Para saber qual o valor a substituir fiz (à parte) um modelo de Regressão Linear

valor_Fare = 7.41689
test.loc[test['Fare'].isna(), 'Fare'] = valor_Fare


test_prediction = pd.get_dummies(test[['Sex', 'Pclass', 'Fare']]) # realiza uma codificação one-hot num DataFrame, usando um conjunto de características (features). A codificação one-hot é uma técnica usada para transformar variáveis categóricas em representações numéricas binárias (0, 1).

final_prediction = RF_GS.predict(test_prediction)


#%% Output do modelo

output_RF_GS_3f1 = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_prediction})
output_RF_GS_3f1.to_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Outputs/RF_GS (3f1).csv', index=False)

