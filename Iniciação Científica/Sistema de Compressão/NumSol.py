import numpy as np
from scipy.optimize import fsolve
import casadi as ca
import plotly.graph_objects as go
import optuna
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.interpolate import griddata

class Simulation:
    def __init__(self, A1, Lc, kv, P1, P_out, C, alphas, N_RotS, nAlphas, nData, perturb,tempo, dt,file_path, decimal=','):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.alphas = alphas
        self.N_RotS = N_RotS
        self.nAlphas = nAlphas
        self.nData = nData
        self.perturb = perturb
        self.dt = dt
        self.tempo = tempo
        
        #Interpolação
        self.file_path = file_path
        self.decimal = decimal
        self.data = None
        self.N_rot = None
        self.Mass = None
        self.Phi = None

        self.interval = [np.linspace(i * self.tempo, (i + 1) * self.tempo, self.nData) for i in range(self.nAlphas)]
        self.time = 0
        
        self.alpha_values = []
        self.N_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.RNN_train = []
        self.RNN_trainFut = []
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        
        self.N_rot = np.arange(2e4,6e4,1e3) # Vai de 20000hz até 60000hz, Shape: (40,)
        self.Mass = np.arange(3,21.1,0.1) # Vai de 3 até 21, Shape: (181,)
        self.Phi = self.data.values # Valores da tabela, Shape: (40,181)
    
    def interpolate(self, num_points=100):
        # Criar uma malha densa para interpolação
        phi_flat = self.Phi.ravel(order='F')
        lut = ca.interpolant('name','bspline',[self.N_rot, self.Mass],phi_flat)

        return lut

    def fun(self, variables, alpha, N):
        (x, y) = variables  # x e y são escalares
        phi_value = float(self.interpolate()([N, x]))  # Garantir que phi_value é escalar
        print(f'Phi: {phi_value}, N: {N}, Mass: {x}, Pressure: {y}, Alpha: {alpha}')
        eqn_1 = (self.A1 / self.Lc) * ((phi_value * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]


    def run(self):
        self.load_data()
        lut = self.interpolate()
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.alphas[0],self.N_RotS[0]))
        print(result)
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        p = ca.MX.sym('p', 2)  # Parâmetros (alpha e N)
        alpha, N = p[0], p[1]  # Divisão dos parâmetros

        # Solução Numérica
        tm1 = time.time()
        for i in range(self.nAlphas):
            alpha_value = self.alphas[i] + np.random.normal(0, self.perturb, self.nData)
            N_value = self.N_RotS[i] + np.random.normal(0, 100, self.nData)
            print(N_value)
            self.alpha_values.append(alpha_value)
            self.N_values.append(N_value)

            rhs = ca.vertcat((self.A1 / self.Lc) * ((lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                             (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))
            
            ode = {'x': x, 'ode': rhs, 'p': p}

            F = ca.integrator('F', 'idas', ode, self.interval[0][0],self.dt)

            for j in range(self.nData):
                params = [alpha_value[j], N_value[j]]
                sol = F(x0=[init_m, init_p], p=params)
                xf_values = np.array(sol["xf"])
                aux1, aux2 = xf_values
                self.massFlowrate.append(aux1)
                self.PlenumPressure.append(aux2)
                init_m = aux1[-1]
                init_p = aux2[-1]
                self.RNN_train.append([aux1[0], aux2[0], alpha_value[j], N_value[j]])
                self.RNN_trainFut.append([aux1[0], aux2[0], alpha_value[j], N_value[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        self.massFlowrate = np.reshape(self.massFlowrate, [self.nAlphas, self.nData])
        self.PlenumPressure = np.reshape(self.PlenumPressure, [self.nAlphas, self.nData])

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
nAlphas = 5 # Número de vezes que o Alfa irá mudar, considere o treino e os testes.
alphas = np.random.uniform(0.35,0.65, nAlphas+1) # Abertura da válvula
N_RotS = np.random.uniform(27e3, 6e4, nAlphas+1)
epochs = 100
nData = 3000
nDataTeste = nData//nAlphas
perturb = 1e-4
tempo = 300
dt = 0.1 # Tempo amostral

# Variáveis auxiliares
interval = [np.linspace(i * tempo, (i + 1) * tempo, nData) for i in range(nAlphas)]
interval_test = [np.linspace(i * tempo, (i + 1) * tempo, nDataTeste) for i in range(nAlphas)]
massFlowrate = []
PlenumPressure = []
alpha_values = []
RNN_train = []
RNN_trainFut = []

# Crie uma instância da classe Simulation
sim = Simulation(A1, Lc, kv, P1, P_out, C, alphas, N_RotS, nAlphas, nData, perturb, tempo, dt, 'E:/Faculdade/UFBA/UFBA/Iniciação Científica/Sistema de Compressão/tabela_phi.csv')

# Carregar os dados necessários
sim.load_data()

sim.run()

print(sim.time)