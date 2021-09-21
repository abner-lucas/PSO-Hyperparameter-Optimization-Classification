import numpy as np

"""Treinar a rede neural usando mini-batch stochastic gradient descent.
   O `training_data` é uma lista de tuplas `(x, y)` representando as entradas 
   de treinamento e as saídas. Os outros parâmetros não opcionais são auto-explicativos. 
   Se `test_data` for fornecido, então a rede será avaliada em relação aos dados 
   do teste após cada época e progresso parcial impresso. Isso é útil para 
   acompanhar o progresso, mas retarda as coisas substancialmente."""

def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [x] 

        # Lista para armazenar todos os vetores z, camada por camada
        zs = [] 

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Aqui, l=1 significa a última camada de neurônios, l=2 é a segunda ... 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def cost_derivative(output_activations, y):
   """Retorna o vetor das derivadas parciais."""
   return (output_activations-y)

# Função de Ativação Sigmóide
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))