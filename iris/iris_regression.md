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

# iris_regression

## Imports

```python
import pandas as pd
```

## Dataset loading and preparation

```python
df = pd.read_csv('Iris.csv')
df.head()
```

```python
transform_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df_transformed = df.copy()
df_transformed['Species'] = df['Species'].map(transform_map)
df_transformed.head()
```

## Regression

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
```

### Train-Test split

```python
from sklearn.model_selection import StratifiedShuffleSplit

X = df_transformed.drop('Species', axis=1)
y = df_transformed['Species']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=47)
split = sss.split(X, y)
for train_index, test_index in split:
    train_index = train_index
    test_index = test_index

X_train = X.iloc[train_index]
y_train = y.iloc[train_index]

X_test = X.iloc[test_index]
y_test = y.iloc[test_index]
```

### Fit and predict

```python
lin_reg.fit(X_train, y_train)
train_predictions = lin_reg.predict(X_test)
```

### Transformation function

```python
def back_transform(number):
    if number <= 0.6999:
        return 0
    elif number <= 1.3999:
        return 1
    else:
        return 2
```

### Calculating accuracy

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score 
train_predictions = pd.Series(train_predictions)
predictions_corrected = train_predictions.map(back_transform)
accuracy = accuracy_score(y_test, predictions_corrected)
recall = recall_score(y_test, predictions_corrected, labels=[0, 1, 2], average=None)
precision = precision_score(y_test, predictions_corrected, labels=[0, 1, 2], average=None)

print(f'Acuracia do modelo: {accuracy}')
print(f'Recall do modelo: {recall}')
print(f'Precisao do modelo: {precision}')
```

## Shuffling y

```python
new_transform_map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 0}
scnd_df = df.copy()
scnd_df['Species'] = df['Species'].map(new_transform_map)

X = scnd_df.drop('Species', axis=1)
y = scnd_df['Species']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=47)
split = sss.split(X, y)
for train_index, test_index in split:
    train_index = train_index
    test_index = test_index

X_train = X.iloc[train_index]
y_train = y.iloc[train_index]

X_test = X.iloc[test_index]
y_test = y.iloc[test_index]

lin_reg.fit(X_train, y_train)
train_predictions = lin_reg.predict(X_test)

train_predictions = pd.Series(train_predictions)
predictions_corrected = train_predictions.map(back_transform)
accuracy = accuracy_score(y_test, predictions_corrected)
recall = recall_score(y_test, predictions_corrected, labels=[1, 2, 0], average=None)
precision = precision_score(y_test, predictions_corrected, labels=[1, 2, 0], average=None)

print(f'Acuracia do modelo: {accuracy}')
print(f'Recall do modelo: {recall}')
print(f'Precisao do modelo: {precision}')
```

Percebemos que ao mudar a ordem dos numeros que codificam nossas classes,
a precisao do modelo muda. Isso provavelmente acontece pela natureza dos dados,
que ao serem descritas por um modelo de regressão linear, não necessariamente seguem
uma lógica crescente, acompanhando a linha de regressão.
