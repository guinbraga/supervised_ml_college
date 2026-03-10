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

# abalone

## Imports

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
## Exploratory data analysis

```python
df = pd.read_csv('abalone.data.csv')
df.head()
```

```python
df.info()
```

```python
df['gender'].value_counts().plot.bar()
```

About equal amounts of each gender. Though what is the I
gender?


```python
df['Rings'].plot.hist(title='Rings distribution')
```
 Really unequal distribution of age. Perhaps log is good encoder? Square-root?

```python
df.hist(bins=50)
```

### Look for correlations

```python
from pandas.plotting import scatter_matrix
numeric = df.columns[df.columns != 'gender']
scatter_matrix(df[numeric], figsize=(12, 8))
```

## Train-Test Split

```python
train, test = train_test_split(df, test_size=0.2, random_state=47)
X_train = train.drop('Rings', axis=1)
y_train = train['Rings']
```

## Prepare data for ML models

### Encode str data

```python
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder()
gender_1hot = oh_encoder.fit_transform(train[['gender']])
gender_1hot
```

### Squashing tails

```python
from sklearn.preprocessing import FunctionTransformer
sqrt_transformer = FunctionTransformer(np.sqrt)
sqrt_height = log_transformer.transform(train['Height'])
sqrt_height.hist()
```

### Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("sqrt", sqrt_transformer),
    ('standardize', StandardScaler())
])

cat_pipeline = Pipeline([
    ('1hot', OneHotEncoder())
])

num_attribs = list(train.columns[train.columns != 'gender'])
cat_attribs = ['gender']

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

train_prepared = preprocessing.fit_transform(train)
train_prepared.shape
```


## Model selection and fitness

### Linear Regression

```python
lr = LinearRegression().fit(X_train, y_train)
```
