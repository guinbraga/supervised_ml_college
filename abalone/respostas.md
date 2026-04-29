# Perguntas

## Qual modelo apresentou o melhor desempenho e porque?

R: A métrica de erro utilizada foi o Erro Quadrático Médio.

O modelo de Árvore de Regressão apresentou menor erro empírico (0.0),
porém maior erro em um dataset desconhecido (3). Isso sugere que o modelo 
se ajustou demais aos dados de treino.

O modelo de regressão linear simples teve um erro empírico próximo
de 2, e o mesmo no dataset desconhecido. Isso sugere que podemos confiar
em sua estimativa de erro quadrático médio próxima de 2, sugerindo um melhor
desempenho do modelo.

## Qual modelo é mais confiável para estimar a taxa de erros para exemplos não conhecidos?

R: Como o modelo de regressão linear teve a menor taxa de erro no dataset de
treino, podemos confiar mais nele para estimar exemplos não conhecidos.
