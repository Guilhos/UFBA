from scipy.optimize import fsolve
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Constantes
alpha0 = np.random.uniform(0.2, 0.8)
A1 = (2.6)*(10**-3)
Vp = 2
Lc = 2
kv = 0.38
P1 = 4.5
P_out = 5
C = 479

# Sistema de equações
def fun(variables, A1, Vp, Lc, kv, P1, P_out, C) :
    (x,y) = variables
    eqn_1 = (A1/Lc)*((1.5 * P1) - y)
    eqn_2 = (C**2)/Vp * (x - alpha0 * kv * np.sqrt(y - P_out))
    return [eqn_1, eqn_2]

# Condições Iniciais
result = fsolve(fun, (0, 10), args = (A1, Vp, Lc, kv, P1, P_out, C)) 
mFlowRate_init = result[0] # Mass Flow inicial
plenumPressure_init = result[1] # Plenum Pressure inicial

# Intervalos no tempo
interval = [np.linspace(i * 400, (i + 1) * 400, 400) for i in range(5)]

# Váriaveis CasADi
x = ca.MX.sym('x', 2) # Variáveis de Estado
alpha = ca.MX.sym('alpha', 1) # Parâmetro Alpha

# Listas para armazenar resultados
mFlowRate_values = []
plenumPressure_values = []
alpha_values = [np.full(400, alpha0)]

# Integrar para cada Intervalo
for i in range(0,5):  
    if i == 0:
        alpha1 = alpha0
    else:
        alpha1 = np.random.uniform(0.2, 0.8)
        alpha_values.append(np.full(400, alpha1))

    rhs = ca.vertcat((A1/Lc)*((1.5 * P1) - x[1]), (C**2)/Vp * (x[0] - alpha * kv * np.sqrt(x[1] - P_out)))
    ode = {'x' : x, 'ode' : rhs, 'p' : alpha }

    F = ca.integrator('F','idas', ode, interval[i][0], interval[i])

    sol = F(x0 = [mFlowRate_init, plenumPressure_init], p = alpha1)

    xf_values = np.array(sol["xf"])
    mFlowRate_values.append(xf_values[0])
    plenumPressure_values.append(xf_values[1])

    # Atualizar as condições iniciais
    mFlowRate_init = xf_values[0][-1]
    plenumPressure_init = xf_values[1][-1]


''' PLOT '''
# Cria uma figura com subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Gráfico da Taxa de Fluxo vs. Tempo
for i in range(0, 5):
    axs[0].plot(interval[i], np.squeeze(mFlowRate_values[i]))
axs[0].grid(True)
axs[0].set_xlabel('Time')  
axs[0].set_ylabel('Mass Flow Rate')
axs[0].set_title('Mass Flow Rate vs Time')
axs[0].legend()

# Gráfico da Pressão do Plenum vs. Tempo
for i in range(0, 5):
    axs[1].plot(interval[i], np.squeeze(plenumPressure_values[i]))
axs[1].grid(True)
axs[1].set_xlabel('Time')  
axs[1].set_ylabel('Plenum Pressure')
axs[1].set_title('Plenum Pressure vs Time')
axs[1].legend()

# Gráfico dos Valores de Alpha vs. Tempo
for i in range(0, 5):
    axs[2].plot(interval[i], np.squeeze(alpha_values[i]), linestyle=':')
axs[2].grid(True)
axs[2].set_xlabel('Time') 
axs[2].set_ylabel('Alpha Value')
axs[2].set_title('Alpha vs Time')
axs[2].legend()

# Ajusta o layout e exibe os gráficos
plt.tight_layout()
plt.show()