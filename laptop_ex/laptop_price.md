---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
---

# laptop_price

## Load and Import

```python
import pandas as pd

train = pd.read_csv('laptop_price_train.csv')
test = pd.read_csv('laptop_price_test.csv')
```

```python
train.head()
```

Vou retirar Product, ja que eh uma coluna pouco informativa

```python
train.drop('Product', axis=1, inplace=True)
```

## Exploration and planning

O DecisionTree do sklearn nao aceita dados nao numericos. Vamos separar os 
dados para encodar os dados categoricos.

```python
train.columns
```

```python
train.info()
```

### Separando colunas em numericas e nao numericas

Verificaremos se todos os pesos estao em 'kg'

```python
(train['Weight'].apply(lambda x: x[-2:]) != 'kg').sum()
```

Como estao, vamos transformar a coluna em numerica

```python
train['Weight'] = train['Weight'].apply(lambda x: x[:-2]).astype('float64')
```

Faremos o mesmo para RAM:

```python
(train['Ram'].apply(lambda x: x[-2:]) != 'GB').sum()
```

```python
train['Ram'] = train['Ram'].apply(lambda x: x[:-2]).astype('int64')
```

```python
cat_features = train.select_dtypes('str').columns.tolist()
cat_features
```

### Investigando a cardinalidade das variaveis numericas

```python
cat_values = train[cat_features].agg(pd.Series.unique)
cat_df = pd.DataFrame(cat_values, columns=['Categories'])
cat_df['n_categories'] = cat_df['Categories'].apply(len)
cat_df.drop('Categories', axis=1, inplace=True)
cat_df
```

Produto eh nossa variavel de maior cardinalidade, mas acredito tambem que seria
interessante dropar essa coluna. Vamos testar o gridsearch com e sem ela. Tambem
vamos separar as variaveis categoricas entre variaveis de alta e baixa
cardinalidade.

```python
baixa_card = ['OpSys', 'TypeName']
alta_card = [cat for cat in cat_features if cat not in baixa_card] 
alta_card
```


## GridSearch

### Definindo o pipeline

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV

card_transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), baixa_card),
    (TargetEncoder(), alta_card),
    remainder='passthrough'
)

card_pipeline = make_pipeline(
    card_transformer,
    DecisionTreeRegressor()
)

tree_params = {
    'decisiontreeregressor__max_depth': [3, 4, 6, 8, 16, None],
    'decisiontreeregressor__min_samples_split': [2, 4, 8, 12],
    'decisiontreeregressor__min_samples_leaf': [2, 4, 8, 12]
}

card_search = GridSearchCV(
    card_pipeline,
    param_grid=tree_params,
    scoring='neg_root_mean_squared_error',
    cv=5
)

X_train = train.drop("Price_euros", axis=1)
y_train = train["Price_euros"]

card_search.fit(X_train, y_train)
```

```python
card_search.best_params_
```


```python
test = pd.read_csv('laptop_price_test.csv')
test['Weight'] = test['Weight'].apply(lambda x: x[:-2]).astype('float64')
test['Ram'] = test['Ram'].apply(lambda x: x[:-2]).astype('int64')
modelo = card_search.best_estimator_
result = modelo.predict(test)
pd.Series(result).to_csv('predict_results.csv')
```

