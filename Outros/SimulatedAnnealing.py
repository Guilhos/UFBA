import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

def func(x):
    return (x+10)*(x+6)*(x+5)*(x+1)*(x-7)*(x-10)/10000

def resfr(t, alpha):
    t = alpha*t
    return t   

def viz(x,dk,a):
    x = x+dk*a
    return x

xx = np.linspace(-11,11,600)

plt.plot(xx, func(xx))

dk = 1

t0 = 80
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
        if delta < 0:
            x = x1
        else:  
            b = rng.random()
            temp = b - np.exp(-delta/t)
            if temp < 0:
                x = x1
        
        if func(x) < func(best):
            best = func(x)

        sols.append(x)
        bests.append(best)
    
    t = resfr(t,alpha)

plt.scatter([best, sols[-1]],[func(best),func(sols[-1])], c = ['r','g'])
# Verde é o ultimo ponto visitado
# Vermelho é o melhor ponto encontrado



