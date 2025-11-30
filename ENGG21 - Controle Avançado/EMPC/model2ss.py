from libs.eos_database import *
from libs.compressor_class import *
from libs.compression import *
from libs.gc_eos_soave import *
from libs.viscosity import *
from libs.duto_casadi import *
from libs.simulation_duto import *
import casadi as ca
from scipy.stats import qmc
import control as ctrl
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class linDiscretizeComp():
    def __init__(self):
        self.dt = 600
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
        gas_temp = self.gas.copy_change_conditions(T0[-1], None, V0[-1], 'gas')
        x0 = np.empty(self.meu_sistema.n_points * 3)
        x0[0::3] = T0
        x0[1::3] = V0
        x0[2::3] = w0
        x0 = np.array(x0, dtype=float)
        z_sol = np.array([np.float64(315.2537249764181), np.float64(0.5140220753802803), np.float64(314.5709878533226), np.float64(0.5286213067888118), np.float64(325.80262286526374), np.float64(0.4200226196815633), np.float64(331.1148887181076), np.float64(0.4290915096513539), np.float64(0.583790379032516)])
        MM = gas_temp.mixture.MM_m  
        v_kg = z_sol[7] / MM
        rho = 1 / v_kg
        z0 = np.hstack((z_sol, [rho * np.pi * (self.D / 2)**2 * w0[0], gas_temp.P]))
        u0 = np.array([600, 4000, 300 , (w0[-1] * np.pi * (self.D / 2)**2) ])

        self.nX = x0.shape[0]
        self.nZ = z0.shape[0]
        self.nU = u0.shape[0]
        self.nY = 3 # Apenas 3 saídas algébricas: Temperatura na saída do compressor, Vazão mássica no compressor e Pressão no final do duto

        self.Cz = np.zeros((self.nY, self.nZ))
        self.outputIndex = [-5,-2,-1]
        for i in range(self.nY):
            self.Cz[i,self.outputIndex[i]] = 1

        self.Cx = np.eye(self.nX)

        self.plant = SimuladorDuto(self.meu_sistema, dt=self.dt, n_steps=1)

        pickle_filename = "C:/Users/guilh/OneDrive/Documents/Workspace/UFBA/ENGG21 - Controle Avançado/EMPC/libs/estEstacionario.pkl"

        if os.path.exists(pickle_filename):
            # Se o arquivo existir, carrega as variáveis
            with open(pickle_filename, 'rb') as f:
                data = pickle.load(f)
            
            self.x_ss = data['x_ss']
            self.z_ss = data['z_ss']
            self.y_ss = data['y_ss']
            self.u_ss = data['u_ss']
            self.SysD = data['SysD']

        else:
            # Se não existir, executa os cálculos e salva os resultados
            self.x_ss, self.z_ss = x0, z0
            self.x_ss,  self.z_ss = self.x_ss.reshape(-1,1), self.z_ss.reshape(-1,1)
            self.y_ss = np.array([self.z_ss[i] for i in self.outputIndex])
            self.u_ss = array(u0).reshape(-1,1)
            self.SysD = self.discretize(self.linearize())

            # Agrupa as variáveis em um dicionário para salvar
            data_to_save = {
                'x_ss': self.x_ss,
                'z_ss': self.z_ss,
                'y_ss': self.y_ss,
                'u_ss': self.u_ss,
                'SysD': self.SysD
            }

            # Salva o dicionário no arquivo pickle
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data_to_save, f)

    def linearize(self):
        x_sym = SX.sym('x', 3*self.meu_sistema.n_points)
        z_sym = SX.sym('z', 11)
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
        Cc = np.block([[-self.Cz @ np.linalg.inv(Azz) @ Azx]]) # Caso queira o X como saída colocar np.block([[self.Cx], [-self.Cz @ np.linalg.inv(Azz) @ Azx]])
        Dc = np.block([[-self.Cz @ np.linalg.inv(Azz) @ Bz]])  # e nessa linha colocar np.block([[np.zeros((self.nX, self.nU))], [-self.Cz @ np.linalg.inv(Azz) @ Bz]])      

        return Ac, Bc, Cc, Dc

    def discretize(self, linSys):
        Ac, Bc, Cc, Dc = linSys

        # Normalização
        Dx = numpy.diag(self.x_ss.flatten())
        Du = numpy.diag(self.u_ss.flatten())
        Dy = numpy.diag(self.y_ss.flatten())

        A_ = numpy.linalg.inv(Dx) @ Ac @ Dx
        B_ = numpy.linalg.inv(Dx) @ Bc @ Du
        C_ = numpy.linalg.inv(Dy) @ Cc @ Dx
        D_ = numpy.linalg.inv(Dy) @ Dc @ Du

        sys_c = ctrl.ss(A_,B_,C_,D_)
        sys_d = ctrl.c2d(sys_c,self.dt, method = 'zoh')
        A, B, C, D = ctrl.ssdata(sys_d)

        # Incremental:
        Atil = np.block([[A,B],[np.zeros((self.nU,self.nX)), np.eye(self.nU)]])
        Btil = np.block([[B],[np.eye(self.nU)]])
        Ctil = np.block([[C, D]])
        Dtil = D.copy()

        return Atil, Btil, Ctil, Dtil, A, B, C, D
    
    def normalize(self, var, type = 'x'):
        var = var.reshape(-1,1)
        if type == 'x':
            var_ = var / self.x_ss - 1
        elif type == 'u':
            var_ = var / self.u_ss - 1
        elif type == 'y':
            var_ = var / self.y_ss - 1
        return var_.reshape(-1,1)
    
    def denormalize(self, var, type = 'x'):
        var = var.reshape(-1,1)
        if type == 'x':
            var_ = self.x_ss * var + self.x_ss
        elif type == 'u':
            var_ = self.u_ss * var + self.u_ss
        elif type == 'y':
            var_ = self.y_ss * var + self.y_ss
        return var_.reshape(-1,1)
    
if __name__ == "__main__":
    # --- Passo 1: Obtenção do sistema discretizado ---
    model = linDiscretizeComp()
    A, B, C, D = model.SysD

    # --- Passo 2: Configuração da Simulação ---
    N_sim = 60*20
    
    Y_hist = np.zeros((C.shape[0], N_sim)) # Histórico para as SAÍDAS
    
    # Condição inicial de estado (X)
    X_k = model.normalize(model.x_ss, 'x')
    
    # Define a entrada de degrau (U)
    u_step = model.normalize(model.u_ss, 'u')
    u_aux = model.u_ss.copy()
    
    # --- Passo 3: Execução do Loop de Simulação ---
    x_k_dev = X_k - model.normalize(model.x_ss, 'x') # Desvio inicial é zero
    u_k_dev = u_step - model.normalize(model.u_ss, 'u') # Desvio da entrada é o degrau
    for k in range(N_sim):
        # Equação de saída (calcula a saída no tempo k)
        # y_dev[k] = C * x_dev[k] + D * u_dev[k]
        y_k_dev = C @ x_k_dev + D @ u_k_dev
        
        # Armazena a saída real (desvio + ponto de operação)
        # Assumindo y_ss = x_ss, já que C=I e D=0
        Y_hist[:, k] = model.denormalize(y_k_dev, 'y').flatten()
        
        # Equação de estados (calcula o próximo estado)
        # x_dev[k+1] = A * x_dev[k] + B * u_dev[k]
        x_k_dev = A @ x_k_dev + B @ u_k_dev
        if k == 301:
            u_aux[0] = 700
            u_aux[3] = 1.92
            u_step = model.normalize(u_aux, 'u')
            u_k_dev = u_step - model.normalize(model.u_ss, 'u')
        elif k == 601:
            u_aux[0] = 670
            u_step = model.normalize(u_aux, 'u')
            u_k_dev = u_step - model.normalize(model.u_ss, 'u')
        elif k == 901:
            u_aux[0] = 710
            u_step = model.normalize(u_aux, 'u')
            u_k_dev = u_step - model.normalize(model.u_ss, 'u')

    # 3.1. Simulação do modelo fenomenológico
    x_f, z_f, u_f = model.x_ss.flatten(), model.z_ss.flatten(), model.u_ss.flatten()

    sim = SimuladorDuto(model.meu_sistema, dt=model.dt, n_steps=N_sim)
    resultados = sim.run(x_f, z_f, u_f)
    T_sol = resultados["T_sol"].T
    w_sol = resultados["w_sol"].T
    V_sol = resultados["V_sol"].T
        
    # --- Passo 4: CRIAR 3 IMAGENS SEPARADAS ---

    # 4.1. Criar o vetor de tempo
    time_vec = np.linspace(0, (N_sim - 1) * model.dt, N_sim)
    time_vec_hours = time_vec / 3600

    # 4.2. Reorganizar os dados por tipo de variável
    n_points = model.meu_sistema.n_points  # Número de pontos espaciais (21)

    # Separar temperaturas, volumes e velocidades do modelo de espaço de estados
    temperaturas_ss = Y_hist[0:62:3, :]    # T0, T1, T2, ..., T20
    volumes_ss = Y_hist[1:62:3, :]         # V0, V1, V2, ..., V20  
    velocidades_ss = Y_hist[2:63:3, :]     # w0, w1, w2, ..., w20

    # Calcular MSE
    mse_temperaturas = np.mean((temperaturas_ss - T_sol)**2, axis=1)
    mse_volumes = np.mean((volumes_ss - V_sol)**2, axis=1)  
    mse_velocidades = np.mean((velocidades_ss - w_sol)**2, axis=1)

    mse_total_temperatura = np.mean(mse_temperaturas)
    mse_total_volume = np.mean(mse_volumes)
    mse_total_velocidade = np.mean(mse_velocidades)

    # Definir o destino para salvar as imagens (altere para o seu caminho desejado)
    destino_imagens = "C:/Users/guilh/OneDrive/Documents/Workspace/UFBA/ENGG21 - Controle Avançado/MPC_Posicional/graficos/"

    # Criar a pasta se não existir
    os.makedirs(destino_imagens, exist_ok=True)

    # 4.3. IMAGEM 1: TEMPERATURAS
    fig1, (ax1_ss, ax1_fen) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle('Comparação de Temperaturas: Modelo de Espaço de Estados vs Fenomenológico', fontsize=14)

    # Espaço de Estados (em cima)
    for i in range(temperaturas_ss.shape[0]):
        ax1_ss.plot(time_vec_hours, temperaturas_ss[i, :], alpha=0.7, linewidth=1)
    ax1_ss.set_ylabel('Temperatura / K')
    ax1_ss.set_title(f'Modelo de Espaço de Estados - MSE: {mse_total_temperatura:.2e}')
    ax1_ss.grid(True)
    #ax1_ss.legend([f'T{i}' for i in range(n_points)], loc='upper right', fontsize=8, ncol=3)

    # Fenomenológico (em baixo)
    for i in range(T_sol.shape[0]):
        ax1_fen.plot(time_vec_hours, T_sol[i, :], alpha=0.7, linewidth=1)
    ax1_fen.set_ylabel('Temperatura / K')
    ax1_fen.set_title('Modelo Fenomenológico')
    ax1_fen.grid(True)
    ax1_fen.set_xlabel('Tempo / h')
    #ax1_fen.legend([f'T{i}' for i in range(n_points)], loc='upper right', fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Salvar a imagem
    plt.savefig(os.path.join(destino_imagens, 'temperaturas_comparacao.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(destino_imagens, 'temperaturas_comparacao.pdf'), bbox_inches='tight')
    plt.show()

    # 4.4. IMAGEM 2: VOLUMES
    fig2, (ax2_ss, ax2_fen) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle('Comparação de Volumes Específicos: Modelo de Espaço de Estados vs Fenomenológico', fontsize=14)

    # Espaço de Estados (em cima)
    for i in range(volumes_ss.shape[0]):
        ax2_ss.plot(time_vec_hours, volumes_ss[i, :], alpha=0.7, linewidth=1)
    ax2_ss.set_ylabel(r'Volume Específico / m³/kmol')
    ax2_ss.set_title(f'Modelo de Espaço de Estados - MSE: {mse_total_volume:.2e}')
    ax2_ss.grid(True)
    #ax2_ss.legend([f'V{i}' for i in range(n_points)], loc='upper right', fontsize=8, ncol=3)

    # Fenomenológico (em baixo)
    for i in range(V_sol.shape[0]):
        ax2_fen.plot(time_vec_hours, V_sol[i, :], alpha=0.7, linewidth=1)
    ax2_fen.set_ylabel(r'Volume Específico / m³/kmol')
    ax2_fen.set_title('Modelo Fenomenológico')
    ax2_fen.grid(True)
    ax2_fen.set_xlabel('Tempo / h')
    #ax2_fen.legend([f'V{i}' for i in range(n_points)], loc='upper right', fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Salvar a imagem
    plt.savefig(os.path.join(destino_imagens, 'volumes_comparacao.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(destino_imagens, 'volumes_comparacao.pdf'), bbox_inches='tight')
    plt.show()

    # 4.5. IMAGEM 3: VELOCIDADES
    fig3, (ax3_ss, ax3_fen) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig3.suptitle('Comparação de Velocidades: Modelo de Espaço de Estados vs Fenomenológico', fontsize=14)

    # Espaço de Estados (em cima)
    for i in range(velocidades_ss.shape[0]):
        ax3_ss.plot(time_vec_hours, velocidades_ss[i, :], alpha=0.7, linewidth=1)
    ax3_ss.set_ylabel('Velocidade / m/s')
    ax3_ss.set_title(f'Modelo de Espaço de Estados - MSE: {mse_total_velocidade:.2e}')
    ax3_ss.grid(True)
    #ax3_ss.legend([f'w{i}' for i in range(n_points)], loc='upper right', fontsize=8, ncol=3)

    # Fenomenológico (em baixo)
    for i in range(w_sol.shape[0]):
        ax3_fen.plot(time_vec_hours, w_sol[i, :], alpha=0.7, linewidth=1)
    ax3_fen.set_ylabel('Velocidade / m/s')
    ax3_fen.set_title('Modelo Fenomenológico')
    ax3_fen.grid(True)
    ax3_fen.set_xlabel('Tempo / h')
    #ax3_fen.legend([f'w{i}' for i in range(n_points)], loc='upper right', fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Salvar a imagem
    plt.savefig(os.path.join(destino_imagens, 'velocidades_comparacao.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(destino_imagens, 'velocidades_comparacao.pdf'), bbox_inches='tight')
    plt.show()

    print(f"Imagens salvas em: {destino_imagens}")