import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# Carregar dados
data = pd.read_csv('recipeData.csv')  # Substitua pelo seu arquivo

# Separar variáveis independentes (X) e dependentes (y)
X = data.drop('target', axis=1)  # Substitua 'target' pelo nome da variável alvo
y = data['target']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tratamento de valores ausentes com imputação
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalização dos dados
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Definir o modelo KNN
knn = KNeighborsClassifier()

# Definir os parâmetros para GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Realizar busca em grade para encontrar os melhores parâmetros
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Melhor modelo
best_knn = grid_search.best_estimator_

# Predição e avaliação
y_pred = best_knn.predict(X_test_scaled)
print(f'Erro Quadrático Médio: {mean_squared_error(y_test, y_pred):.4f}')
