---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: .venv (3.11.12)
    language: python
    name: python3
---

## Carregar pacotes

```python
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
```

## Carregar conjunto de dados Iris

```python
dataset_iris = pd.read_csv('../data/Iris.csv')
dataset_iris
```

## Remover a coluna `Id`

```python
dataset_iris.drop(columns=['Id'], inplace=True)
dataset_iris
```

## Gráfico da disposição dos exemplos do conjunto Iris em dois eixos

```python
fig, ax = plt.subplots()
ax.scatter(dataset_iris['SepalLengthCm'], dataset_iris['PetalWidthCm'], c=pd.factorize(dataset_iris['Species'])[0])


plt.show()
```

# Dividir o conjunto de dados em treino e teste
## 80% treino e 20% teste

```python
splitter = ShuffleSplit(n_splits=1, test_size=.20)
strat_splits = []
for train_index, test_index in splitter.split(dataset_iris):
    train_set_n = dataset_iris.iloc[train_index]
    test_set_n = dataset_iris.iloc[test_index]
    strat_splits.append([train_set_n, test_set_n])


train_set = strat_splits[0][0]
test_set = strat_splits[0][1]

print('Total de exemplos')
print(f'Treino: {len(train_set)}')
print(f'Teste: {len(test_set)}')
```

## Gráfico da disposição dos exemplos dos conjuntos de treino

```python
dataset = train_set

fig, ax = plt.subplots()
ax.scatter(dataset['SepalLengthCm'], dataset['SepalWidthCm'], c=[pd.factorize(dataset['Species'])[0]])


plt.show()
```

## Gráfico da disposição dos exemplos dos conjuntos de teste

```python
dataset = test_set

fig, ax = plt.subplots()
ax.scatter(dataset['SepalLengthCm'], dataset['SepalWidthCm'], c=pd.factorize(dataset['Species'])[0])


plt.show()
```

## Subdivide os conjuntos de treino e teste em conjuntos de características e conjunto de rótulos

```python
X_train_set = train_set.drop(columns=['Species'])
y_train_set = train_set['Species'].copy()

X_test_set = test_set.drop(columns=['Species'])
y_test_set = test_set['Species'].copy()
```

## Define arquitetura da Árvore de Decisão e realiza o treinamento

```python
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=None)
dtc.fit(X_train_set, y_train_set)
```

## Cria uma representação gráfica do modelo gerado pelo método de árvore de decisão

### A representação é salva em dois arquivos:
### - iris_tree.dot
### - iris_tree.png

```python
export_graphviz(
    dtc,
    out_file='iris_tree.dot',
    feature_names=X_train_set.columns,
    rounded=True,
    filled=True
)

! dot -Tpng iris_tree.dot -o iris_tree.png
```

## Exibe o arquivo iris_tree.png

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('iris_tree.png')
plt.figure(figsize=(10, 8))
imgplot = plt.imshow(img)
plt.show()
```

## Realiza a predição do conjunto de teste e exibe o desempenho

### As métricas de desempenho utilizadas são: acurácia, revocação e precisão

```python
predicts_dtc = dtc.predict(X_test_set)


print(f'Acurácia: {accuracy_score(y_test_set, predicts_dtc)}')
print(f'Revocação: {recall_score(y_test_set, predicts_dtc, average=None)}')
print(f'Precisão: {precision_score(y_test_set, predicts_dtc, average=None)}')
```

<br><br><br><br>

# Exercício

## Altere a estrutura da árvore de decisão utilizada e, para cada configuração, replique pelo menos 3 vezes (gerando novos conjuntos de treino e teste) e indique a menor arquitetura encontrada. 

### Definindo Parametros e funcoes para repitibilidade:

```python
MAX_DEPTHS = (2, 3, 4, 5, None)

def split(splitter: ShuffleSplit, dataset: pd.DataFrame) -> tuple:
    strat_splits = []
    for train_index, test_index in splitter.split(dataset):
        train_set_n = dataset.iloc[train_index]
        test_set_n = dataset.iloc[test_index]
        strat_splits.append([train_set_n, test_set_n])


    train_set = strat_splits[0][0]
    test_set = strat_splits[0][1]

    return (train_set, test_set)
```

```python
import numpy as np
resultados = np.zeros(len(MAX_DEPTHS))
for n_split in range(3):
    train_set, test_set = split(splitter, dataset_iris)
    X_train_set = train_set.drop(columns=['Species'])
    y_train_set = train_set['Species'].copy()

    X_test_set = test_set.drop(columns=['Species'])
    y_test_set = test_set['Species'].copy()

    for i, depth in enumerate(MAX_DEPTHS):
        dtc = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        dtc.fit(X_train_set, y_train_set)
        predicts_dtc = dtc.predict(X_test_set)
        
        resultados[i] += accuracy_score(y_test_set, predicts_dtc)

resultados /= 3
pd.Series(resultados, name='Acuracia', index=MAX_DEPTHS)
```

Pudemos ver, portanto, que as melhores arvores sao as de tamanho 3 e 5,
e nao a arvore sem limite de tamanho
