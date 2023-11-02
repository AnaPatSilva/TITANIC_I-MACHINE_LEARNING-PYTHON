# -*- coding: utf-8 -*-
"""
Usar os datasets do Titanic, que se encontram no Kaggle, para prever os falecidos do naufrágio
Para treinar
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

# Vou usar apenas a variável Sex
train['Sex'].value_counts() #Verificar a distribuição do sexo no dataset
train['Sex'].value_counts(normalize=True)

#%%Divisão dos dados e definição da Feature (Sex) e do Target/Label (Survived)

#y = target
y_train_target = train['Survived']

#x = feature
x_feature = ['Sex']
x_train_feature = pd.get_dummies(train[x_feature]) # realiza uma codificação one-hot num DataFrame, usando um conjunto de características (features). A codificação one-hot é uma técnica usada para transformar variáveis categóricas em representações numéricas binárias (0, 1).
x_test_feature = pd.get_dummies(test[x_feature])

#%% Treinamento do modelo inicial de Random Forest

from sklearn.ensemble import RandomForestClassifier

#n_estimators (nº de árvores) = 100
#max_depth (quantidade de camadadas) = 5
#random_state (seed) = 42
RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
RF.fit(x_train_feature, y_train_target)
y_prediction_target = RF.predict(x_test_feature)


#%% Medição da precisão do modelo inicial

# O ideal seria comparar as previsões com o dataset de teste, mas este não contém o valor do target
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test_target, y_prediction_target)
#print(f'Precisão do modelo: {accuracy:.2f}')


#%% Cross Validation para evitar o over fitting

from sklearn.model_selection import cross_val_score, KFold

# número de dobras = 5
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
scores_kf = cross_val_score(RF, x_train_feature, y_train_target, cv=kf, scoring='accuracy')
print("Precisão para cada fold:", ["%.2f" % score for score in scores_kf])
print("Precisão média: %.2f" % scores_kf.mean())


from sklearn.model_selection import cross_val_score, RepeatedKFold

# número de repetições = 3
num_repeats = 3
rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
scores_rkf = cross_val_score(RF, x_train_feature, y_train_target, cv=rkf, scoring='accuracy')
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
random_search.fit(x_train_feature, y_train_target)

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
grid_search.fit(x_train_feature, y_train_target)
melhores_hiperparametros_gs = grid_search.best_params_

# Obter o melhor modelo encontrado pelo GridSearch
melhor_modelo_gs = grid_search.best_estimator_

# Avaliar o melhor modelo nos dados de teste
#acuracia_teste = melhor_modelo.score(x_test_feature, y_test) #não consigo medir a accuracy porque não tenho o target do test

# Obtenha a melhor pontuação (score)
melhor_pontuacao_gs = grid_search.best_score_

# Exibir os melhores hiperparâmetros e a acurácia no conjunto de teste
print("Melhores Hiperparâmetros:", melhores_hiperparametros_gs)
print("Melhor Estimador (Modelo):", melhor_modelo_gs.get_params())
print("Melhor Pontuação (Score): %.2f" % melhor_pontuacao_gs)
#print("Acurácia no Conjunto de Teste:", acuracia_teste)


#%% Treinamento do modelo de Random Forest ajustado (tuned)
#Como ambos os tuning tiveram o mesmo resultado vou treinar com ambos os hiperparametros

RF_RS = RandomForestClassifier(
    n_estimators=melhores_hiperparametros_rs['n_estimators'],
    max_depth=melhores_hiperparametros_rs['max_depth'],
    min_samples_split=melhores_hiperparametros_rs['min_samples_split'],
    min_samples_leaf=melhores_hiperparametros_rs['min_samples_leaf'],
    random_state=42  # Defina a semente aleatória se necessário
)
RF_RS.fit(x_train_feature, y_train_target)
y_prediction_target_rs = RF_RS.predict(x_test_feature)


RF_GS = RandomForestClassifier(
    n_estimators=melhores_hiperparametros_gs['n_estimators'],
    max_depth=melhores_hiperparametros_gs['max_depth'],
    min_samples_split=melhores_hiperparametros_gs['min_samples_split'],
    min_samples_leaf=melhores_hiperparametros_gs['min_samples_leaf'],
    random_state=42  # Defina a semente aleatória se necessário
)
RF_GS.fit(x_train_feature, y_train_target)
y_prediction_target_gs = RF_GS.predict(x_test_feature)


#%% Medição da precisão do modelo após o ajuste

# O ideal seria comparar as previsões com o dataset de teste, mas este não contém o valor do target
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(test_target, prediction_target)
#print(f'Precisão do modelo: {accuracy:.2f}')


# Assim vou fazer uma curva ROC

from sklearn.metrics import roc_curve, roc_auc_score

target_probs_GS = RF_GS.predict_proba(x_test_feature)
#Calcular as taxas de verdadeiro positivo (TPR) e as taxas de falso positivo (FPR) usando as probabilidades previstas e os rótulos reais
#Vou calcular as taxas usando o Sexo Feminino como o mais provável de ter sobrevivido
#target_probs[:, 0] = O : antes da vírgula significa "selecione todas as linhas" e o 0 após a vírgula significa "selecione a primeira coluna"
fpr_GS, tpr_GS, thresholds_GS = roc_curve(x_test_feature['Sex_female'], target_probs_GS[:, 0])

# Calcular a área sob a curva ROC (AUC-ROC)
auc_roc_GS = roc_auc_score(x_test_feature['Sex_female'], target_probs_GS[:, 0])

# Plote da curva ROC

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr_GS, tpr_GS, color='blue', lw=2, label=f'AUC = {auc_roc_GS:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curva ROC para a classe "Sex_female')
plt.legend(loc='lower right')
plt.show()
print("AUC-ROC:", auc_roc_GS)
# A curva não está a funcionar


#%% Matriz de Confusão
#É necessário o y_test_target, mas não temos essa informação

#from sklearn.metrics import confusion_matrix


#%% Resumo dos resultados
print("Precisão média Cross Folding: %.2f" % scores_kf.mean()) #0,79
print("Precisão média Repetated Cross Folding: %.2f" % scores_rkf.mean()) #0,79
print("Melhor Pontuação (Score) Random Search: %.2f" % melhor_pontuacao_rs) #0,79
print("Melhor Pontuação (Score) Grid Search: %.2f" % melhor_pontuacao_gs) #0,79


#%% Output do modelo

output_RF_GS_1f = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_prediction_target_gs})
output_RF_GS_1f.to_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Outputs/RF_GS (1f).csv', index=False)
