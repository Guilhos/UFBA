from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# Método FSolve SCIPY
def ECW(f,e,D,Re):
  X = -2*np.log10(e/(D*3.7) + 2.51/Re*f**0.5)*f**0.5 - 1
  return X

xx = np.linspace(0,0.3,100)

e = 1.5e-4
D = 5e-2
Re = 1e5

res = fsolve(ECW,0,args=(e,D,Re))

print(res)

fig, (axs1,axs2) = plt.subplots(1,2,figsize=(10,4))

axs1.plot(xx,ECW(xx,e,D,Re))
axs1.plot(res,ECW(res,e,D,Re),'o')
axs1.set_title("Fator de Atrito: Colebrook-White")
axs1.grid(True, which='both')

# Método da Bisecção
def EVW(Vm,P,T,R,a,b):
  Y = R*T - (P + a/Vm**2)*(Vm-b)
  return Y

a,b = -0.2,0.6
xx = np.linspace(a,b,100)

P = 5000 #kPa
T = 300
R = 8.314
a = 3.59
b = 4.27*1e-5
epsilon = 1e-9
x1 = np.random.uniform(a,b,1)
x2 = np.random.uniform(a,b,1)
f1 = EVW(x1,P,T,R,a,b)
f2 = EVW(x2,P,T,R,a,b)

for j in range(100):
    if f1*f2 > 0:
        print(f"Intervalo entre {x1} e {x2} não é válido, pois não contém a raíz")
        x1 = np.random.uniform(a,b)
        x2 = np.random.uniform(a,b)
        f1 = EVW(x1,P,T,R,a,b)
        f2 = EVW(x2,P,T,R,a,b)
    else:
        N = int(np.log(np.abs(x2-x1)/epsilon)/np.log(2))
        for i in range(N):
            x3 = 0.5*(x1+x2)
            f3 = EVW(x3,P,T,R,a,b)

            if f3 >= -epsilon and f3 <= epsilon:
                print(x3)
                break

            if f2*f3 < 0:
                x1 = x3
                f1 = f3
            else:
                x2 = x3
                f2 = f3
        break

axs2.plot(xx,EVW(xx,P,T,R,a,b))
axs2.plot(x3,f3,'o')
axs2.set_title("Volume Molar: Van der Waals")
axs2.grid(True, which='both')

plt.show()