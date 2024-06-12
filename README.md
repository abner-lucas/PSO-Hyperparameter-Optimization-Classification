# Rede Neural Profunda com Otimização de Hiperparâmetros usando PSO
Este projeto utiliza uma rede neural profunda para classificação no conjunto de dados Iris, com otimização de hiperparâmetros (pesos e bias) através de Particle Swarm Optimization (PSO). Inclui análise exploratória de dados e dois casos experimentais no conjunto de dados Iris (carregado através da biblioteca Scikit-learn), contendo 150 amostras de três espécies de flores (Setosa, Virginica, Versicolor) com quatro características (comprimento e largura das sépalas e pétalas). Foram conduzidos dois testes experimentais para otimizar os hiperparâmetros da rede neural profunda.

## Dataset
Para os casos experimentais, o conjunto de dados foi dividido com 70% das amostras para treinamento e 30% para teste, garantindo a aleatoriedade das amostras.

## Rede Neural
Uma rede neural perceptron de três camadas foi criada: uma camada de entrada, uma camada oculta com 20 neurônios e uma camada de saída com 3 neurônios para indicar as três espécies de flores. O método de propagação direta (feedforward) e o algoritmo PSO foram utilizados para otimizar os hiperparâmetros da rede, evoluindo os valores de peso e bias ao longo das épocas de treinamento.

## Casos Experimentais
Foram definidos valores de parâmetros específicos para a execução do algoritmo de PSO durante o treinamento da rede neural. Dois casos experimentais foram conduzidos:

### Caso 1
Trabalhado com as quatro características das flores para treinar a rede neural. O PSO foi configurado com 100 partículas e 1000 épocas, resultando em uma acurácia de treino de 100% e uma acurácia de teste de 91,11%.

### Caso 2
Utilizado apenas o comprimento e largura das pétalas para treinar a rede neural. O PSO foi configurado da mesma forma que no Caso 1, resultando em uma acurácia de treino de 99,05% e uma acurácia de teste de 97,78%.

## Conclusão
A otimização por enxame de partículas mostrou-se eficaz para automatizar a seleção de valores de hiperparâmetros de uma rede neural profunda, alcançando uma alta probabilidade de resultados assertivos. O uso de características relevantes do conjunto de dados pode aumentar a precisão do processo de classificação.

## Requisitos
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `pyswarms`

## Referências
- [Scikit-learn](https://scikit-learn.org/)
- [PySwarms](https://pyswarms.readthedocs.io/)

## Autor
- [@abner-lucas](https://github.com/abner-lucas)
  
## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo [MIT](https://choosealicense.com/licenses/mit/) para mais detalhes
