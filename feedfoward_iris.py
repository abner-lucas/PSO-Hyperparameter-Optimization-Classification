import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pyswarms as ps

# Carregando dataset de iris
dataset = load_iris()

# Guardando dados de entrada e as saídas desejadas para cada amostra de treinamento
amostras = dataset.data[:130]
y_desejado = dataset.target[:130]

# Arquiteura da rede neural
n_entrada = 4                # número de entradas na camada de entrada
n_oculta = 20                # número de neurônios na camada oculta
n_saida = 3                  # número de classes na camada de saídas
n_amostras = len(amostras)   # número de amostras

def logits_function(p):
    # Construindo vetores iniciais de pesos e bias a partir da lista de parâmetros
    W1 = p[0:80].reshape((n_entrada,n_oculta))      # pesos oculta-entrada
    b1 = p[80:100].reshape((n_oculta,))             # bias oculta-entrada
    W2 = p[100:160].reshape((n_oculta,n_saida))     # pesos saída-oculta
    b2 = p[160:163].reshape((n_saida,))             # bias saída-oculta

    # Ativando camadas
    z1 = amostras.dot(W1) + b1      # nets da camada oculta
    a1 = np.tanh(z1)                # ativação da camada oculta
    y_obtido = a1.dot(W2) + b2      # nets da camada de saída
    return y_obtido                 # saídas da rede neural

# Propagação direta da rede neural
def forward_prop(particula):
    y_obtido = logits_function(particula)

    # Ativação da camada de saída
    exp_scores = np.exp(y_obtido)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calcula o erro da saída de rede
    corect_logprobs = -np.log(probs[range(n_amostras), y_desejado]) # erro por saída
    erro = np.sum(corect_logprobs) / n_amostras                     # erro médio total

    return erro

def swarm(enxame):
    n_particulas = enxame.shape[0]
    enxame = [forward_prop(enxame[i]) for i in range(n_particulas)]
    return np.array(enxame)

# Definindo fator de aceleração 1 e 2 e peso de inércia para o PSO
parametrosPSO = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Iniciando PSO
dimensoes = (n_entrada * n_oculta) + (n_oculta * n_oculta) + n_oculta + n_oculta
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensoes, options=parametrosPSO)
# Realizando treinamento/otimização
fitness, posicao = optimizer.optimize(swarm, iters=3)

# Calculando a acurácia
def predict(posicao):
    logits = logits_function(posicao)
    y_obtido = np.argmax(logits, axis=1)
    return y_obtido

# Acurácia do treino
print((predict(posicao) == y_desejado).mean())

# Guardando dados de entrada e as saídas desejadas para cada amostra de teste
amostras = dataset.data[130:]
y_desejado = dataset.target[130:]

# Acurácia do teste
print((predict(posicao) == y_desejado).mean())