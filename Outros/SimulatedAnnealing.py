import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# Função exemplo Aula
def func(x):
    return (x+10)*(x+6)*(x+5)*(x+1)*(x-7)*(x-10)/10000

# Função de Italo
# def func(x):
#     if x < -1 or x > 1:
#         y = 0
#     else:
#         y = (np.cos(50*x) + np.sin(20*x))
#     return y
# fv = np.vectorize(func)

xx = np.linspace(-11,11,500)
plt.plot(xx, func(xx))

def resfr(t, alpha):
    t = alpha*t
    return t   

def viz(x,dk,a):
    x = x+dk*a
    return x

dk = 1
t0 = 60
alpha = 0.75
x0 = -11
n1 = 20
n2 = 50
sols = []
bests = []

x = x0
t = t0
best = x

for i in range(n1):
    for j in range(n2):
        a = rng.random() * 2 - 1
        x1 = viz(x,dk,a)
        delta = func(x1) - func(x)
        if x1 > -11 and x1 < 11:
            if delta < 0:
                x = x1
            else:  
                b = rng.random()
                temp = b - np.exp(-delta/t)
                if temp < 0:
                    x = x1
            
            if func(x) < func(best):
                best = x

            sols.append(x)
            bests.append(best)
    
    t = resfr(t,alpha)

plt.plot(sols, [func(y) for y in sols], c = 'gray')
plt.scatter(bests, [func(y) for y in bests], c = 'black')
plt.scatter([x0, best, sols[-1]],[func(x0),func(best),func(sols[-1])], c = ['b','r','g'])
# Verde é o ultimo ponto visitado
# Vermelho é o melhor ponto encontrado



