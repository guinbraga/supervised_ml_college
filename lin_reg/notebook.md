---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Regressão Linear

```python
from pathlib import Path
import numpy as np
import pandas as pd
import tarfile
import urllib.request
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import Function
```

<br><br>

## Baixar e carregar o dataset

```python

def load_housing_data():
    tarball_path = Path("dataset/housing.tgz")
    if not tarball_path.is_file():
        Path("dataset").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="dataset")
    return pd.read_csv(Path("dataset/housing/housing.csv"))

housing = load_housing_data()
```

<br><br>

## Visualização rápida dos dados

```python
housing.head()
```

```python
housing.info()
```

```python
housing["ocean_proximity"].value_counts()
```

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix
```

```python
housing.columns
```

<br><br>

## Seleciona algumas colunas para um novo conjunto de dados

```python
housing_cut = housing.drop(columns=['longitude', 'latitude', 'total_bedrooms', 'households', 'ocean_proximity'])
```

```python
housing_cut
```

<br><br>

## Divide o conjunto de dados em treino (80%) e teste (20%)

```python
splitter = ShuffleSplit(n_splits=1, test_size=.20, random_state=0)
strat_splits = []
for train_index, test_index in splitter.split(housing_cut):
    train_set_n = housing_cut.iloc[train_index]
    test_set_n = housing_cut.iloc[test_index]
    strat_splits.append([train_set_n, test_set_n])


train_set = strat_splits[0][0]
test_set = strat_splits[0][1]

print('Total de exemplos')
print(f'Treino: {len(train_set)}')
print(f'Teste: {len(test_set)}')
```

```python
X_train_set = train_set.drop(columns=['median_house_value'])
y_train_set = train_set['median_house_value'].copy()

X_test_set = test_set.drop(columns=['median_house_value'])
y_test_set = test_set['median_house_value'].copy()
```

<br><br>

## Treino

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train_set, y_train_set)
```

```python
lin_reg.intercept_
```

<br><br>

## Teste

```python
values_predictions = lin_reg.predict(X_test_set)
```

<br><br>

## Análise

```python
lin_mae = mean_absolute_error(y_test_set, values_predictions)
print(f'Erro médio absoluto: {round(lin_mae, 2)}')
```

### 1. Qual o erro absoluto médio no conjunto de treinamento?

```python
train_predictions = lin_reg.predict(X_train_set)
train_mae = mean_absolute_error(train_predictions, y_train_set)
print(f'Erro médio absoluto no conjunto de treinamento: {round(train_mae, 2)}')
```

### 2. Qual a diferença (absoluta) entre o erro no conjunto de treinamento e teste?

```python
diff = np.abs(train_mae - lin_mae)
print(f'A diferença abosluta entre os erros no conjunto de treinamento e teste é de {diff:.8}')
```

### 3. Demonstre o percentual da diferença entre os erros nos conjuntos de treino e teste.

```python
perc_diff = round((lin_mae-train_mae)/train_mae*100, 2)
print(f'A diferença percentual do erro médio absoluto no conjunto de treinamento e no conjunto de teste foi de {perc_diff}%')
```

### 4. Modifique o código visto e explore alternativas em busca de alcançar melhor desempenho (menor erro).

Vamos verificar se ao aplicar transformações nas features do dataset, temos um melhor desempenho.
Primeiro, vamos verificar a distribuição das variáveis numericas:

```python
housing.hist(bins=30)
```

 Verificamos que temos várias variáveis com uma cauda longa, então
vamos aplicar log a elas para suavizar, e depois normalizar. 
Tambem vamos imputar nos valores nulos de total_bedrooms

```python
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
)

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

transformation = ColumnTransformer([
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population',
                           'households', 'median_income']),

    ], remainder=num_pipeline
)

housing_selected = housing.drop(['longitude', 'latitude', 'ocean_proximity'], axis=1)
```
#### Dividindo entre treino e teste

```python
splitter = ShuffleSplit(n_splits=1, test_size=.20, random_state=0)
strat_splits = []
for train_index, test_index in splitter.split(housing_selected):
    train_set_n = housing_selected.iloc[train_index]
    test_set_n = housing_selected.iloc[test_index]
    strat_splits.append([train_set_n, test_set_n])


train_set = strat_splits[0][0]
test_set = strat_splits[0][1]

print('Total de exemplos')
print(f'Treino: {len(train_set)}')
print(f'Teste: {len(test_set)}')

X_train_set = train_set.drop(columns=['median_house_value'])
y_train_set = train_set['median_house_value'].copy()

X_test_set = test_set.drop(columns=['median_house_value'])
y_test_set = test_set['median_house_value'].copy()
```

#### Testando

```python
lin_reg = make_pipeline(transformation, LinearRegression())
lin_reg.fit(X_train_set, y_train_set)
```

```python
train_predictions = lin_reg.predict(X_train_set)
train_mae = mean_absolute_error(train_predictions, y_train_set)
print(f'O erro medio absoluto no conjunto de treino foi de {round(train_mae, 2)}')
```

```python
test_predictions = lin_reg.predict(X_test_set)
test_mae = mean_absolute_error(test_predictions, y_test_set)
print(f'O erro medio absoluto no conjunto de teste foi de {round(test_mae, 2)}')
```

Ao utilizarmos mais variaveis, imputarmos valores nulos com a mediana e aplicarmos
transformacoes, reduzimos o erro medio absoluto.
