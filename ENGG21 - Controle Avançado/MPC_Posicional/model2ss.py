from libs.eos_database import *
from libs.compressor_class import *
from libs.compression import *
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.plenum_system import *
from libs.duto_casadi import *
import casadi as ca
from scipy.stats import qmc
import control as ctrl
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class linDiscretizeComp():
    def __init__(self):
        self.list_names = ["CH4", "C2H6", "C3H8", "iC4H10", "nC4H10", "iC5H12", "nC5H12", 
                  "nC6H14", "nC7H16", "nC8H18", "nC9H20", "nC10H22", "nC11H24", 
                   "nC12H26", "nC14H30", "N2", "H2O", "CO2", "C15+"]

        self.nwe = [0.9834, 0.0061, 0.0015, 0.0003, 0.0003, 0.00055, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0008, 0.0]

        self.dict_composition= {self.list_names[i]: self.nwe[i] for i in range(len(self.nwe))}

        self.mixture = Mixture(list_of_species, self.dict_composition)

        self.volumn_desviation = [0] * 19

        self.Vpp = 2.0 
        self.linearizeA1 = 2.6e-3
        self.gas = gc_eos_class(self.mixture, 300, 4000, None, 1, 0, Aij, self.volumn_desviation, 'gas')
        visc = viscosity(self.mixture, self.volumn_desviation)
        self.D = 0.42
        # Criar instância do duto com os parâmetros necessários  # ajuste conforme seu import real

        # Número de nós (comprimento de self.l)

        self.comp = CompressorClass()
        self.compressor = compression(self.gas, self.comp, visc)

        self.meu_sistema = duto_casadi(self.gas, visc, compressor=self.compressor, Lc=100000.0, D=0.42)
        # --- Condições iniciais ---
        T0 = [331.11488872, 331.03855994, 330.88678422, 330.66133717, 330.36492314, 330.00125445, 329.57511687, 329.0924697, 328.56054495, 327.98795287, 327.38477399, 326.76262096, 326.13464356, 325.51544488, 324.92086969, 324.36762591, 323.87270874, 323.45261962, 323.12241303, 322.89464857, 322.77838627]
        V0 = [0.42909151, 0.43015709, 0.4322893, 0.4354898, 0.43975979, 0.44509723, 0.45149414, 0.45893212, 0.46737692, 0.47677125, 0.48702615, 0.49801102, 0.50954311, 0.52137775, 0.53320164, 0.54463245, 0.5552283, 0.56451055, 0.57200092, 0.57727063, 0.57999387]
        w0 = [6.08904527, 6.1041665, 6.13442364, 6.1798406, 6.2404341, 6.31617523, 6.40695091, 6.51249996, 6.6323363, 6.76564704, 6.91116972, 7.06705115, 7.2306979, 7.3986379, 7.56642544, 7.7286349, 7.87899586, 8.01071621, 8.11700869, 8.19178876, 8.23043304]
        x0 = np.empty(self.meu_sistema.n_points * 3)
        x0[0::3] = T0
        x0[1::3] = V0
        x0[2::3] = w0
        x0 = np.array(x0, dtype=float)
        z0 = np.array([np.float64(315.2537249764181), np.float64(0.5140220753802803), np.float64(314.5709878533226), np.float64(0.5286213067888118), np.float64(325.80262286526374), np.float64(0.4200226196815633), np.float64(331.1148887181076), np.float64(0.4290915096513539), np.float64(0.583790379032516)])
        u0 = np.array([600, 4000, 300 , (w0[-1] * np.pi * (self.D / 2)**2) ])
        pickle_filename = "C:/Users/guilh/OneDrive/Documents/Workspace/UFBA/ENGG21 - Controle Avançado/MPC_Posicional/libs/estEstacionario.pkl"

        if os.path.exists(pickle_filename):
            # Se o arquivo existir, carrega as variáveis
            with open(pickle_filename, 'rb') as f:
                data = pickle.load(f)
            
            self.x_ss = data['x_ss']
            self.z_ss = data['z_ss']
            self.u_ss = data['u_ss']
            self.SysD = data['SysD']

        else:
            # Se não existir, executa os cálculos e salva os resultados
            self.x_ss, self.z_ss = x0, z0
            self.x_ss,  self.z_ss = self.x_ss.reshape(-1,1), self.z_ss.reshape(-1,1)
            self.u_ss = array(u0).reshape(-1,1)
            self.SysD = self.discretize(self.linearize())

            # Agrupa as variáveis em um dicionário para salvar
            data_to_save = {
                'x_ss': self.x_ss,
                'z_ss': self.z_ss,
                'u_ss': self.u_ss,
                'SysD': self.SysD
            }

            # Salva o dicionário no arquivo pickle
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data_to_save, f)

    def linearize(self):
        x_sym = SX.sym('x', 3*self.meu_sistema.n_points)
        z_sym = SX.sym('z', 9)
        u_sym = SX.sym('u', 4)
        t = ca.SX.sym("t")

        dydt, alg_eqs = self.meu_sistema.evaluate_dae(t, x_sym, z_sym, u_sym)
        dae = {"x": x_sym, "z": z_sym, "p": u_sym, "ode": dydt, "alg": alg_eqs}

        f_expr = dae['ode']
        g_expr = dae["alg"]

        # EDOs
        Axx_sym = ca.jacobian(f_expr, x_sym) # df/dx
        Axz_sym = ca.jacobian(f_expr, z_sym) # df/dz
        Bx_sym = ca.jacobian(f_expr, u_sym) # df/du

        eval_Axx = ca.Function('eval_Axx', [x_sym, z_sym, u_sym], [Axx_sym])
        eval_Axz = ca.Function('eval_Axz', [x_sym, z_sym, u_sym], [Axz_sym])
        eval_Bx = ca.Function('eval_Bx', [x_sym, z_sym, u_sym], [Bx_sym])

        # ALG
        Azx_sym = ca.jacobian(g_expr, x_sym) # dg/dx
        Azz_sym = ca.jacobian(g_expr, z_sym) # dg/dz
        Bz_sym = ca.jacobian(g_expr, u_sym) # dg/du

        eval_Azx = ca.Function('eval_Axx', [x_sym, z_sym, u_sym], [Azx_sym])
        eval_Azz = ca.Function('eval_Axz', [x_sym, z_sym, u_sym], [Azz_sym])
        eval_Bz = ca.Function('eval_Bx', [x_sym, z_sym, u_sym], [Bz_sym])

        Axx = np.squeeze(eval_Axx(self.x_ss, self.z_ss, self.u_ss))
        Axz = np.squeeze(eval_Axz(self.x_ss, self.z_ss, self.u_ss))
        Bx = np.squeeze(eval_Bx(self.x_ss, self.z_ss, self.u_ss))
        Azx = np.squeeze(eval_Azx(self.x_ss, self.z_ss, self.u_ss))
        Azz = np.squeeze(eval_Azz(self.x_ss, self.z_ss, self.u_ss))
        Bz = np.squeeze(eval_Bz(self.x_ss, self.z_ss, self.u_ss))

        # dotX = Axx @ X + Axz @ Z + Bx @ U
        # 0 = Azx @ X + Azz @ Z + Bz @ U
        # Z = - Azz^{-1} @ Azx @ X - Azz^{-1} @ Bz @ U
        # dotX = (Axx - Axz @ Azz^{-1} @ Azx) @ X + (Bx - Axz @ Azz^{-1} @ Bz) @ U

        Ac = Axx - (Axz @ np.linalg.inv(Azz) @ Azx)
        Bc = Bx - Axz @ np.linalg.inv(Azz) @ Bz

        return Ac, Bc

    def discretize(self, linSys):
        Ac, Bc = linSys

        # Normalização
        Dx = numpy.diag(self.x_ss.flatten())
        Du = numpy.diag(self.u_ss.flatten())
        
        A_ = numpy.linalg.inv(Dx) @ Ac @ Dx
        B_ = numpy.linalg.inv(Dx) @ Bc @ Du

        dt = 0.5
        
        Cc = np.eye(len(Ac))
        Dc = np.zeros((Ac.shape[0], Bc.shape[1]))

        sys_c = ctrl.ss(A_,B_,Cc,Dc)
        sys_d = ctrl.c2d(sys_c,dt, method = 'zoh')
        A, B, C, D = ctrl.ssdata(sys_d)

        print(np.linalg.eigvals(A))

        return A, B, C, D
    
    def normalize(self, var, type = 'x'):
        var = var.reshape(-1,1)
        if type == 'x':
            var_ = var / self.x_ss - 1
        elif type == 'u':
            var_ = var / self.u_ss - 1
        return var_.reshape(-1,1)
    
    def denormalize(self, var, type = 'x'):
        var = var.reshape(-1,1)
        if type == 'x':
            var_ = self.x_ss * var + self.x_ss
        elif type == 'u':
            var_ = self.u_ss * var + self.u_ss
        return var_.reshape(-1,1)
    
if __name__ == "__main__":
    # --- Passo 1: Obtenção do sistema discretizado ---
    model = linDiscretizeComp()
    A, B, C, D = model.SysD

    # --- Passo 2: Configuração da Simulação ---
    N_sim = 80
    dt = 0.5
    
    Y_hist = np.zeros((C.shape[0], N_sim)) # Histórico para as SAÍDAS
    
    # Condição inicial de estado (X)
    X_k = model.normalize(model.x_ss)
    
    # Define a entrada de degrau (U)
    u_step = model.normalize(model.u_ss, 'u')
    u_step[2] = model.normalize(model.u_ss, 'u')[2] + 0.15 # Aumenta a rotação (terceira entrada) em 15%
    
    # --- Passo 3: Execução do Loop de Simulação ---
    x_k_dev = X_k - model.normalize(model.x_ss) # Desvio inicial é zero
    u_k_dev = u_step - model.normalize(model.u_ss, 'u') # Desvio da entrada é o degrau
    for k in range(N_sim):
        # Equação de saída (calcula a saída no tempo k)
        # y_dev[k] = C * x_dev[k] + D * u_dev[k]
        y_k_dev = C @ x_k_dev + D @ u_k_dev
        
        # Armazena a saída real (desvio + ponto de operação)
        # Assumindo y_ss = x_ss, já que C=I e D=0
        Y_hist[:, k] = model.denormalize(y_k_dev).flatten()
        
        # Equação de estados (calcula o próximo estado)
        # x_dev[k+1] = A * x_dev[k] + B * u_dev[k]
        x_k_dev = A @ x_k_dev + B @ u_k_dev
        if k == 40:
            u_step[2] = model.normalize(model.u_ss, 'u')[2] - 0.15
            u_k_dev = u_step - model.normalize(model.u_ss, 'u')

    # x_ss, z_ss = model.x_ss.copy(), model.z_ss.copy()
        
    # x_sym = SX.sym('x', 3)
    # z_sym = SX.sym('z', 11)
    # u_sym = SX.sym('u', 5)

    # ode_sym, alg_sym = model.plenum_sys.evaluate_dae(None, x_sym, z_sym, u_sym)
    # dae = {
    #     'x': x_sym,
    #     'z': z_sym,
    #     'p': u_sym,
    #     'ode': vertcat(*ode_sym),
    #     'alg': vertcat(*alg_sym)
    # }

    # integrator_solver = integrator('F', 'idas', dae, {'tf': dt})

    # time_steps = []
    # x_values = []
    # z_values = []
    # alpha_values = []  
    # N_values = []

    # u0NonLin = model.u0.copy()

    # u0NonLin[2] = u0NonLin[2] * 1.15
    # for j in range(N_sim):
    #     res = integrator_solver(x0=x_ss, z0=z_ss, p=u0NonLin)
        
    #     x_ss = np.array(res["xf"])
    #     z_ss = np.array(res["zf"])
        
    #     x_values.append(x_ss.copy())
    #     z_values.append(z_ss.copy())
    #     if N_sim == 40:
    #         u0NonLin[2] = u0NonLin[2] / 1.15

    # x_values = np.array(x_values).reshape(-1,3)
    # x_values = x_values.T
        
    # --- Passo 4: PLOT com Matplotlib (Gráficos Separados) ---
    
    # 4.1. Criar o vetor de tempo
    time_vec = np.linspace(0, (N_sim - 1) * dt, N_sim)

    # 4.2. Criar a figura e os subplots
    # 3 linhas, 1 coluna. sharex=True liga o eixo X de todos os gráficos.
    fig, axes = plt.subplots(63, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Resposta das Saídas do Sistema a um Degrau na Entrada', fontsize=16)

    #MSE = np.mean(np.square(x_values - Y_hist), axis = 1)

    # Nomes das saídas para os títulos (ajuste conforme o significado de cada uma)
    #output_names = [f'Vazão Mássica, MSE: {MSE[0]}', f'Temperatura, MSE: {MSE[1]}', f'Volume, MSE: {MSE[2]}']

    # 4.3. Loop para plotar cada saída em seu respectivo subplot
    for i in range(Y_hist.shape[0]):
        axes[i].plot(time_vec, Y_hist[i, :], label=f'Resposta de Espaço de Estados')
        #axes[i].plot(time_vec, x_values[i, :], label=f'Resposta do modelo não linear')
        # Adiciona uma linha tracejada para indicar o valor inicial (estado estacionário)
        axes[i].axhline(y=model.x_ss[i], color='r', linestyle='--', label=f'Y$_{i+1}$ (ss)')
        
        #axes[i].set_title(output_names[i])
        axes[i].set_ylabel('Valor')
        axes[i].grid(True)
        axes[i].legend()

    # 4.4. Adicionar rótulo do eixo X apenas no último gráfico
    axes[-1].set_xlabel('Tempo (s)')

    # 4.5. Ajustar o layout e exibir o gráfico
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o supertítulo não sobrepor
    plt.show()
