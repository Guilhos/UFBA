import numpy as np
from scipy.optimize import fsolve
import casadi as ca
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd

class Interpolation:
    def __init__(self, file_path, decimal = ','):
        self.file_path = file_path
        self.decimal = decimal
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        self.N_rot = np.arange(2e4,6e4,1e3) # Vai de 20000hz até 60000hz, Shape: (40,)
        self.Mass = np.arange(3,21.1,0.1) # Vai de 3 até 21, Shape: (181,)
        self.Phi = self.data.values # Valores da tabela, Shape: (40,181)
    
    def interpolate(self):
        # Criar uma malha densa para interpolação
        phi_flat = self.Phi.ravel(order='F')
        lut = ca.interpolant('name','bspline',[self.N_rot, self.Mass],phi_flat)

        return lut 

class Simulation:
    def __init__(self, A1, Lc, kv, P1, P_out, C, N_RotS, nCiclos, nData, perturb, tempo, dt, interpolation):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.N_RotS = N_RotS
        self.nCiclos = nCiclos
        self.nData = nData
        self.perturb = perturb
        self.dt = dt
        self.tempo = tempo
        
        #Interpolação
        self.interpolation = interpolation
        self.data = None
        self.N_rot = None
        self.Mass = None
        self.Phi = None

        self.interval = [np.linspace(i * self.tempo, (i + 1) * self.tempo, self.nData) for i in range(self.nCiclos)]
        self.time = 0
        
        self.N_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.Phi_values = []
        self.RNN_train = []
        self.RNN_trainFut = []
        

    def fun(self, variables, N):
        (x, y) = variables  # x e y são escalares
        phi_value = float(self.interpolation([N, x]))  # Garantir que phi_value é escalar
        eqn_1 = (self.A1 / self.Lc) * ((phi_value * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - 0.5 * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]


    def run(self):
        lut = self.interpolation
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.N_RotS[0]))
        print(result)
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        N = ca.MX.sym('N', 1)

        # Solução Numérica
        tm1 = time.time()
        for i in range(self.nCiclos):
            N_value = self.N_RotS[i] + np.random.normal(0, 50, self.nData)
            self.N_values.append(N_value)

            rhs = ca.vertcat((self.A1 / self.Lc) * ((lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                             (self.C**2) / 2 * (x[0] - 0.5 * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))
            
            ode = {'x': x, 'ode': rhs, 'p': N}

            F = ca.integrator('F', 'cvodes', ode, self.interval[0][0],self.dt)

            for j in range(self.nData):
                params = [N_value[j]]
                sol = F(x0=[init_m, init_p], p=params)
                xf_values = np.array(sol["xf"])
                aux1, aux2 = xf_values
                self.massFlowrate.append(aux1)
                self.PlenumPressure.append(aux2)
                self.Phi_values.append(lut(ca.vertcat(N_value[j], aux1)))
                init_m = aux1[-1]
                init_p = aux2[-1]
                self.RNN_train.append([aux1[0], aux2[0], N_value[j]])
                self.RNN_trainFut.append([aux1[0], aux2[0], N_value[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        self.massFlowrate = np.reshape(self.massFlowrate, [self.nCiclos, self.nData])
        self.PlenumPressure = np.reshape(self.PlenumPressure, [self.nCiclos, self.nData])
        self.Phi_values = np.reshape(self.Phi_values, [self.nCiclos, self.nData])

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
N_RotS = np.random.uniform(27e3, 50e3, nCiclos+1)
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

lut = Interpolation('E:/Faculdade/UFBA/UFBA/Iniciação Científica/Sistema de Compressão/tabela_phi.csv')
lut.load_data()
interpolation = lut.interpolate()
# Crie uma instância da classe Simulation
sim = Simulation(A1, Lc, kv, P1, P_out, C, N_RotS, nCiclos, nData, perturb, tempo, dt, interpolation)
# Execute a simulação
sim.run()

RNN_train = sim.RNN_train
RNN_trainFut = sim.RNN_trainFut
massFlowrate = sim.massFlowrate
PlenumPressure = sim.PlenumPressure
phi_values = sim.Phi_values

print(sim.time)

fig = make_subplots(rows=2, cols=2, subplot_titles=("Vazão vs Tempo", "Pressão vs Tempo", " ", "Phi vs Tempo"))

for i in range(0, nCiclos):
    # Vazão
    fig.add_trace(go.Scatter(x=interval[i], y=np.squeeze(massFlowrate[i]), mode='lines',
                             name='Vazão', legendgroup='massflow', showlegend=i == 0), row = 1, col = 1)
    # Pressão
    fig.add_trace(go.Scatter(x=interval[i], y=np.squeeze(PlenumPressure[i]), mode='lines',
                             name='Pressão', legendgroup='pressure', showlegend=i == 0), row = 1, col = 2)
    # Phi
    fig.add_trace(go.Scatter(x=interval[i], y=np.squeeze(phi_values[i]), mode='lines', 
                             name='Alphas', line=dict(dash='dash'), legendgroup='alpha', showlegend=i == 0), row = 2, col = 2)

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