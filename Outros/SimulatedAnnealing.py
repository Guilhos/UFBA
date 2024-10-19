import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# Função exemplo Aula
def func1(x):
    return (x+10)*(x+6)*(x+5)*(x+1)*(x-7)*(x-10)/10000

# Função de Italo
# def func2(x):
#     if x < -1 or x > 1:
#         y = 0
#     else:
#         y = (np.cos(50*x) + np.sin(20*x))
#     return y
# fv = np.vectorize(func)

xx = np.linspace(-11,11,500)
plt.plot(xx, func1(xx))

dx = 1 # Distância máxima da vizinhança
t0 = 60 # Temperatura inicial
alpha = 0.75 # Constante de redução da Temperatura
x0 = -11 # X inicial
n1 = 20 # Número de temperaturas
n2 = 50 # Número de iterações em cada temperatura

def resfr(t, alpha):
    t = alpha*t
    return t   

def viz(x,dk,a):
    x = x+dk*a
    return x

def SA(F, x0, t0, dx, alpha, n1, n2):
    sols = []
    bests = []

    x = x0
    t = t0
    best = x

    for i in range(n1):
        for j in range(n2):
            a = rng.random() * 2 - 1
            x1 = viz(x,dx,a)
            delta = F(x1) - F(x)
            if x1 > -11 and x1 < 11:
                if delta < 0:
                    x = x1
                else:  
                    b = rng.random()
                    temp = b - np.exp(-delta/t)
                    if temp < 0:
                        x = x1
                
                if F(x) < F(best):
                    best = x

                sols.append(x)
                bests.append(best)
        
        t = resfr(t,alpha)

    return sols, bests

sol, best = SA(func1, x0, t0, dx, alpha, n1, n2)

def plotter(F, sol, best):
    plt.plot(sol, [F(y) for y in sol], c = 'gray')
    plt.scatter(best, [F(y) for y in best], c = 'black')
    plt.scatter([x0, best[-1], sol[-1]],[F(x0),F(best[-1]),F(sol[-1])], c = ['b','r','g'])
# Verde é o ultimo ponto visitado
# Vermelho é o melhor ponto encontrado

plotter(func1, sol, best)