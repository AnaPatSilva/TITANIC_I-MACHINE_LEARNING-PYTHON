# -*- coding: utf-8 -*-
"""
Vou fazer uma Multiple Linear Regression para substituir o valor nulo da feature Fare
"""

#%% Instalação de livrarias
#%pip install scikit-learn
#%pip install seaborn


#%% Importação de datasets

import os
import pandas as pd

for dirname, _, filenames in os.walk('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#gender = pd.read_csv('G:\O meu disco\Formação\Kaggle\Kaggle Titanic\Data\gender_submission.csv')
test = pd.read_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Data/test.csv') # tive de mudar a direção da barra, porque estava a dar erro
#train = pd.read_csv('G:/O meu disco/Formação/Kaggle/Kaggle Titanic/Data/train.csv')

#print(gender.head())
print(test.head())
#print(train.head())

print(test.shape)
test.describe()


#%% Data Profiling do dataset de treino (comentado para não gerar relatório sempre que corro o código)

#from ydata_profiling import ProfileReport

#profile_train = ProfileReport(train, title="Profile Train")
#profile_train.to_file("profile_train.html")


#%% Fazer uma Matriz de Correlação
test_columnsdrop = ['Name', 'Ticket', 'Cabin', 'Embarked'] #colunas a não incluir no dataset para o onehot
test_corr = test.drop(test_columnsdrop, axis=1)
test_onehot = pd.get_dummies(test_corr, columns=['Sex'], drop_first=True)
correlation_matrix = test_onehot.corr()
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

# Criar um mapa de calor da matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

# Obter as correlações da feature 'Fare' com todas as outras features
correlations_target = correlation_matrix['Fare']

# Remover a correlação com a própria feature 'Fare' (que será 1)
correlations_target = correlations_target.drop('Fare')

# Obter as features mais correlacionadas com 'Fare' em ordem decrescente
top_features = correlations_target.abs().sort_values(ascending=False).head(3)

# Imprimir uma lista das features mais correlacionadas com 'Fare' juntamente com seus valores de correlação
print("As features mais correlacionadas com 'Fare':")
for feature_name in top_features.index:
    correlation_value = top_features[feature_name]
    print(f"{feature_name} {correlation_value:.2f}")

#Há uma correlação grande entre Fare e Pclass, Age e Parch

#Ver as linhas com valores nulos
NA = test[test.isna().any(axis=1)]
print(NA)
#só aparecem missing values na coluna da Cabin

# Vou usar as variáveis Pclass, Age e Parch
test['Pclass'].value_counts() #Verificar a distribuição da feature no dataset
test['Pclass'].value_counts(normalize=True)
test['Age'].value_counts()
test['Age'].value_counts(normalize=True)
test['Parch'].value_counts()
test['Parch'].value_counts(normalize=True)

NA_Pclass = test['Pclass'].isnull()
NA_Pclass_sum = NA_Pclass.sum()
print(f'Número de valores nulos em "Pclass": {NA_Pclass_sum}')

NA_Age = test['Age'].isnull()
NA_Age_sum = NA_Age.sum()
print(f'Número de valores nulos em "Age": {NA_Age_sum}')

NA_Parch = test['Parch'].isnull()
NA_Parch_sum = NA_Parch.sum()
print(f'Número de valores nulos em "Parch": {NA_Parch_sum}')


# Como a variável Age tem muitos NA, vou substituir essa variável por outra
top_features = correlations_target.abs().sort_values(ascending=False).head(4)

# Imprimir uma lista das features mais correlacionadas com 'Fare' juntamente com seus valores de correlação
print("As features mais correlacionadas com 'Fare':")
for feature_name in top_features.index:
    correlation_value = top_features[feature_name]
    print(f"{feature_name} {correlation_value:.2f}")
    
#Vou substituir por Sex
NA_Sex = test['Sex'].isnull()
NA_Sex_sum = NA_Sex.sum()
print(f'Número de valores nulos em "Sex": {NA_Sex_sum}')



#%%Divisão dos dados e definição da Feature (Pclass, Parch, Sex) e do Target/Label (Fare)

Titanic_Fare = test[['Fare', 'Pclass']]
#Titanic_Fare = pd.get_dummies(Titanic_Fare, columns=["Sex"],drop_first='False')

# Separe os dados em duas partes: com valores ausentes e completos na variável Target (Fare)
dados_completos = Titanic_Fare.dropna(subset=['Fare'])
dados_ausentes = Titanic_Fare[Titanic_Fare['Fare'].isnull()]

# Separe as variáveis target e feature
x_completos = dados_completos[['Pclass']]
y_completos = dados_completos['Fare']

# Crie um modelo de regressão linear e ajuste-o aos dados completos

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(x_completos, y_completos)

# Preveja os valores ausentes com base nos preditores
x_ausentes = dados_ausentes[['Pclass']]
previsoes_ausentes = LR.predict(x_ausentes)

# Impute os valores previstos na variável alvo
dados_ausentes.loc[dados_ausentes['Fare'].isna(), 'Fare'] = previsoes_ausentes

# Combine os dados completos e imputados
dados_imputados = pd.concat([dados_completos, dados_ausentes])

# Agora 'dados_imputados' contém os valores imputados na variável alvo
print(dados_imputados)


# Vou visualizar os dados do Fare para verificar se o valor imputado faz sentido

print(dados_imputados['Fare'])
