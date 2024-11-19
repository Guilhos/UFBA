import numpy as np
from scipy.optimize import fsolve
import casadi as ca
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class Simulation:
    def __init__(self, A1, Lc, kv, P1, P_out, C, alphas, nCiclos, nData, perturb,tempo, dt):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.alphas = alphas
        self.nCiclos = nCiclos
        self.nData = nData
        self.perturb = perturb
        self.dt = dt
        self.tempo = tempo

        self.interval = [np.linspace(i * self.tempo, (i + 1) * self.tempo, self.nData) for i in range(self.nCiclos)]
        self.time = 0
        
        self.alpha_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.Phi_values = []
        self.RNN_train = []
        self.RNN_trainFut = []
        

    def fun(self, variables, alpha):
        (x, y) = variables  # x e y são escalares
        eqn_1 = (self.A1 / self.Lc) * ((1.5 * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]


    def run(self):
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.alphas[0]))
        print(result)
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        alpha = ca.MX.sym('alpha', 1)  # Parâmetros (alpha e N)

        # Solução Numérica
        tm1 = time.time()
        for i in range(self.nCiclos):
            alpha_value = self.alphas[i] + np.random.normal(0, self.perturb, self.nData)
            self.alpha_values.append(alpha_value)

            rhs = ca.vertcat((self.A1 / self.Lc) * ((1.5 * self.P1) - x[1]) * 1e3,
                             (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))
            
            ode = {'x': x, 'ode': rhs, 'p': alpha}

            F = ca.integrator('F', 'cvodes', ode, self.interval[0][0],self.dt)

            for j in range(self.nData):
                params = [alpha_value[j]]
                sol = F(x0=[init_m, init_p], p=params)
                xf_values = np.array(sol["xf"])
                aux1, aux2 = xf_values
                self.massFlowrate.append(aux1)
                self.PlenumPressure.append(aux2)
                init_m = aux1[-1]
                init_p = aux2[-1]
                self.RNN_train.append([aux1[0], aux2[0], alpha_value[j]])
                self.RNN_trainFut.append([aux1[0], aux2[0], alpha_value[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        self.massFlowrate = np.reshape(self.massFlowrate, [self.nCiclos, self.nData])
        self.PlenumPressure = np.reshape(self.PlenumPressure, [self.nCiclos, self.nData]) 

np.random.seed(42)
print(np.random.seed)

# Constantes
A1 = (2.6)*(10**-3)
Lc = 2
kv = 0.38
P1 = 4.5
P_out = 5
C = 479

timestep = 3 # Passos no passado para prever o próximo
nCiclos = 5 # Número de vezes que o Alfa irá mudar, considere o treino e os testes.
alphas = np.random.uniform(0.35,0.65, nCiclos+1) # Abertura da válvula
epochs = 1500
nData = 600
nDataTeste = nData//nCiclos
perturb = 1e-4
tempo = 60
dt = 0.1 # Tempo amostral

# Variáveis auxiliares
interval = [np.linspace(i * tempo, (i + 1) * tempo, nData) for i in range(nCiclos)]
interval_test = [np.linspace(i * tempo, (i + 1) * tempo, nDataTeste) for i in range(nCiclos)]
massFlowrate = []
PlenumPressure = []
alpha_values = []
RNN_train = []
RNN_trainFut = []

sim = Simulation(A1, Lc, kv, P1, P_out, C, alphas, nCiclos, nData, perturb, tempo, dt)
# Execute a simulação
sim.run()

RNN_train = sim.RNN_train
RNN_trainFut = sim.RNN_trainFut
massFlowrate = sim.massFlowrate
PlenumPressure = sim.PlenumPressure
alpha_values = sim.alpha_values

print(sim.time)

fig = make_subplots(rows=2, cols=2, subplot_titles=("Vazão vs Tempo", "Pressão vs Tempo", "Alpha vs Tempo"))

for i in range(0, nCiclos):
    # Vazão
    fig.add_trace(go.Scatter(x=interval[i], y=np.squeeze(massFlowrate[i]), mode='lines',
                             name='Vazão', legendgroup='massflow', showlegend=i == 0), row = 1, col = 1)
    # Pressão
    fig.add_trace(go.Scatter(x=interval[i], y=np.squeeze(PlenumPressure[i]), mode='lines',
                             name='Pressão', legendgroup='pressure', showlegend=i == 0), row = 1, col = 2)
    # Alphas
    fig.add_trace(go.Scatter(x=interval[i], y=np.squeeze(alpha_values[i]), mode='lines', 
                             name='Alphas', line=dict(dash='dash'), legendgroup='alpha', showlegend=i == 0), row = 2, col = 1)

# Atualiza layout
fig.update_layout(
    xaxis_title='Tempo',
    grid=dict(rows=1, columns=3),
    template='plotly',
    showlegend=False,
    height = 600
)

# Mostra a figura
fig.show()