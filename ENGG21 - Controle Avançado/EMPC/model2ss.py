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
        """
        Para usar os códigos do MPC, essa classe deve conter as seguintes partes principais:
        - O sistema discreto de Espaço de Estados -> Aqui o Sys.D
        - Os pontos estacionários onde o sistema foi linearizado -> Aqui os x_ss, z_ss, u_ss e y_ss
        - As respectivas dimensões das variáveis -> Aqui o nX, nZ, nU e nY
        - O tempo amostral -> Aqui o dt
        - A planta que você irá utilizar -> Aqui o plant
        
        Esse código, da forma que está, apenas funciona caso o sistema seja fenomenológico e compatível com Casadi, mas é possível usar com outras ferramentas,
        porém é necessário mudar a forma de linearização e como você cumpre as necessidades listadas acima
        """
        self.dt = 600

        # SEU MODELO --- Utilizado como exemplo: Compressor + Duto
        ## Obs.: Coloque os arquivos necessários na pasta 'libs', por questão de organização
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
        
        ### Criar instância do duto com os parâmetros necessários  # ajuste conforme seu import real
        ### Número de nós (comprimento de self.l)
        self.comp = CompressorClass()
        self.compressor = compression(self.gas, self.comp, visc)

        self.meu_sistema = duto_casadi(self.gas, visc, compressor=self.compressor, Lc=200000.0, D=0.42)
        
        self.CustoW = 0.00091 # [=] R$/Wh

        # --- Condições iniciais ---
        x0 = np.empty(self.meu_sistema.n_points * 3)

        T0 = [344.38357044, 342.11387519, 337.8789603, 332.21319592, 325.75576282, 319.1194846, 312.79510357, 307.10466433, 302.2078673, 298.1363481, 294.83740597, 292.21489608, 290.156918, 288.55417424, 287.31163613, 286.35221591, 285.61733276, 285.06542981, 284.66905088, 284.41177752, 284.28493562]
        V0 = [0.36277098, 0.36076218, 0.35709748, 0.35241593, 0.34752295, 0.3432504, 0.34033443, 0.33933335, 0.3406009, 0.34429833, 0.35042694, 0.35886151, 0.36936893, 0.38161267, 0.39513791, 0.40934773, 0.42348406, 0.43663143, 0.44777369, 0.45590683, 0.46020776]
        w0 = [5.79141036, 5.75934112, 5.70083653, 5.6260986, 5.54798522, 5.47977671, 5.43322494, 5.4172434, 5.4374791, 5.49650613, 5.59434562, 5.72899811, 5.89674254, 6.09220617, 6.30812808, 6.53497884, 6.76065646, 6.97054598, 7.14842511, 7.27826563, 7.34692732]

        x0[0::3] = T0
        x0[1::3] = V0
        x0[2::3] = w0

        x0 = np.array(x0, dtype=float)

        z_sol = np.array([np.float64(320.96133173352536), np.float64(0.4833304429738393), np.float64(320.3585160045981), np.float64(0.4951168340183638), np.float64(338.89539580027076), np.float64(0.35506519067468956), np.float64(344.3835704373247), np.float64(0.36277097658886426), np.float64(0.583790379032516)])
        gas_temp = self.gas.copy_change_conditions(T0[-1], None, V0[-1], 'gas')
        MM = gas_temp.mixture.MM_m  
        v_kg = z_sol[7] / MM
        rho = 1 / v_kg
        
        u0 = np.array([700])

        self.compressor.compressor.update_speed(u0[0])
        self.potencia_inicial = self.compressor.compressor.Ah_ideal

        z0 = np.hstack((z_sol, [rho * np.pi * (self.D / 2)**2 * w0[0], gas_temp.P, self.potencia_inicial*(rho * np.pi * (self.D / 2)**2 * w0[0])*600/1000*0.000277778]))

        ## As condições iniciais do sistemas são dadas por x0, z0 e u0

        # DIMENSÕES DAS VARIÁVEIS
        self.nX = x0.shape[0]
        self.nZ = z0.shape[0]
        self.nU = u0.shape[0]
        self.nY = 4 # No sistema em exemplo, quero extrair apenas 4 saídas

        # FILTRO DE SAÍDAS DO ESPAÇO DE ESTADOS
        ## Estados diferenciais
        self.Cx = np.eye(self.nX)

        ## Estados algébricos
        self.Cz = np.zeros((self.nY, self.nZ))
        self.outputIndex = [-6,-3,-2,-1]
        for i in range(self.nY):
            self.Cz[i,self.outputIndex[i]] = 1 ## Aqui apenas coleto as saídas algébricas nas posições outputIndex

        ### Obs.: Dependendo da forma que você diz que esses estados serão medidos, você deverá alterar a lógica da matriz C

        # PLANTA DO SEU SISTEMA
        self.plant = SimuladorDuto(self.meu_sistema, dt=self.dt, n_steps=1)

        # ARQUIVO PARA ARMAZENAR O ESPAÇO DE ESTADO E ESTADOS ESTACIONÁRIOS, PARA NÂO FICAR RODANDO O SISTEMA NOVAMENTE TODA VEZ
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
            sim2ss = SimuladorDuto(self.meu_sistema, dt = 600, n_steps= 1000)

            sim2ss.run(x0.flatten(), z0.flatten(), u0.flatten())
            T0ss = sim2ss.resultados['T_sol'][-1,:]
            V0ss = sim2ss.resultados['V_sol'][-1,:]
            w0ss = sim2ss.resultados['w_sol'][-1,:]

            self.x_ss = np.empty(self.meu_sistema.n_points * 3)
            self.x_ss[0::3] = T0ss
            self.x_ss[1::3] = V0ss
            self.x_ss[2::3] = w0ss

            self.x_ss = np.array(self.x_ss, dtype=float).reshape(-1,1)

            self.z_ss = sim2ss.resultados['z_sol'][-1,:].T.reshape(-1,1)

            # Se não existir, executa os cálculos e salva os resultados
            self.y_ss = np.array([self.z_ss[i] for i in self.outputIndex])
            self.u_ss = np.array(u0).reshape(-1,1)
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
        # LINEARIZAÇÂO DO SISTEMA
        x_sym = ca.SX.sym('x', 3*self.meu_sistema.n_points)
        z_sym = ca.SX.sym('z', 12)
        u_sym = ca.SX.sym('u', 1)
        t = ca.SX.sym("t")

        ## Sistema simbólico
        dydt, alg_eqs = self.meu_sistema.evaluate_dae(t, x_sym, z_sym, u_sym)
        dae = {"x": x_sym, "z": z_sym, "p": u_sym, "ode": dydt, "alg": alg_eqs}

        f_expr = dae['ode']
        g_expr = dae["alg"]

        # Série de Taylor de Primeira Ordem
        ## EDOs
        Axx_sym = ca.jacobian(f_expr, x_sym) # df/dx
        Axz_sym = ca.jacobian(f_expr, z_sym) # df/dz
        Bx_sym = ca.jacobian(f_expr, u_sym) # df/du

        eval_Axx = ca.Function('eval_Axx', [x_sym, z_sym, u_sym], [Axx_sym])
        eval_Axz = ca.Function('eval_Axz', [x_sym, z_sym, u_sym], [Axz_sym])
        eval_Bx = ca.Function('eval_Bx', [x_sym, z_sym, u_sym], [Bx_sym])

        ## ALG
        Azx_sym = ca.jacobian(g_expr, x_sym) # dg/dx
        Azz_sym = ca.jacobian(g_expr, z_sym) # dg/dz
        Bz_sym = ca.jacobian(g_expr, u_sym) # dg/du

        eval_Azx = ca.Function('eval_Axx', [x_sym, z_sym, u_sym], [Azx_sym])
        eval_Azz = ca.Function('eval_Axz', [x_sym, z_sym, u_sym], [Azz_sym])
        eval_Bz = ca.Function('eval_Bx', [x_sym, z_sym, u_sym], [Bz_sym])

        # Matrizes no ponto de estado estacionário
        Axx = np.squeeze(eval_Axx(self.x_ss, self.z_ss, self.u_ss))
        Axz = np.squeeze(eval_Axz(self.x_ss, self.z_ss, self.u_ss))
        Bx = np.squeeze(eval_Bx(self.x_ss, self.z_ss, self.u_ss)).reshape(-1,1)
        Azx = np.squeeze(eval_Azx(self.x_ss, self.z_ss, self.u_ss))
        Azz = np.squeeze(eval_Azz(self.x_ss, self.z_ss, self.u_ss))
        Bz = np.squeeze(eval_Bz(self.x_ss, self.z_ss, self.u_ss)).reshape(-1,1)

        # dotX = Axx @ X + Axz @ Z + Bx @ U
        # 0 = Azx @ X + Azz @ Z + Bz @ U
        # Z = - Azz^{-1} @ Azx @ X - Azz^{-1} @ Bz @ U
        # dotX = (Axx - Axz @ Azz^{-1} @ Azx) @ X + (Bx - Axz @ Azz^{-1} @ Bz) @ U

        # Espaço de Estados no contínuo posicional
        Ac = Axx - (Axz @ np.linalg.inv(Azz) @ Azx)
        Bc = Bx - Axz @ np.linalg.inv(Azz) @ Bz
        Cc = np.block([[-self.Cz @ np.linalg.inv(Azz) @ Azx]]) # Caso queira os Estados Diferenciais (X) como saída colocar np.block([[self.Cx], [-self.Cz @ np.linalg.inv(Azz) @ Azx]])
        Dc = np.block([[-self.Cz @ np.linalg.inv(Azz) @ Bz]])  # e nessa linha colocar np.block([[np.zeros((self.nX, self.nU))], [-self.Cz @ np.linalg.inv(Azz) @ Bz]])      

        return Ac, Bc, Cc, Dc

    def discretize(self, linSys):
        # DISCRETIZAÇÃO DO SISTEMA
        Ac, Bc, Cc, Dc = linSys

        ## Normalização - Da forma que está -> Normaliza valores para [-1,1], sendo o ss -> 0, 2*ss -> 1, 0 -> -1. Dessa forma, colocar em desvio fica mais simples.
        Dx = np.diag(self.x_ss.flatten())
        Du = np.diag(self.u_ss.flatten())
        Dy = np.diag(self.y_ss.flatten())

        A_ = np.linalg.inv(Dx) @ Ac @ Dx
        B_ = np.linalg.inv(Dx) @ Bc @ Du
        C_ = np.linalg.inv(Dy) @ Cc @ Dx
        D_ = np.linalg.inv(Dy) @ Dc @ Du

        ## Discretização do Espaço de Estados Normalizados
        sys_c = ctrl.ss(A_,B_,C_,D_)
        sys_d = ctrl.c2d(sys_c,self.dt, method = 'zoh')
        A, B, C, D = ctrl.ssdata(sys_d)

        ## Incremental:
        Atil = np.block([[A, B], [np.zeros((self.nU, self.nX)), np.eye(self.nU)]])
        Btil = np.block([[B], [np.eye(self.nU)]])
        Ctil = np.block([[C, D]])
        Dtil = D.copy()

        return Atil, Btil, Ctil, Dtil, A, B, C, D

    # FUNÇÕES AUXILIARES
    def normalize(self, var, type = 'x'): # Normalizar entre [-1,1]
        var = var.reshape(-1,1)
        if type == 'x':
            var_ = var / self.x_ss - 1
        elif type == 'u':
            var_ = var / self.u_ss - 1
        elif type == 'y':
            var_ = var / self.y_ss - 1
        return var_.reshape(-1,1)
    
    def denormalize(self, var, type = 'x'): # Desnormalizar de [-1,1]
        var = var.reshape(-1,1)
        if type == 'x':
            var_ = self.x_ss * var + self.x_ss
        elif type == 'u':
            var_ = self.u_ss * var + self.u_ss
        elif type == 'y':
            var_ = self.y_ss * var + self.y_ss
        return var_.reshape(-1,1)
    
    def get_output_names(self):
        """Retorna os nomes das saídas"""
        return ['y0', 'y1', 'y2', 'y3']

# FUNÇÃO PARA SIMULAR SISTEMA INCREMENTAL
def simulate_incremental_system(model, delta_u_sequence):
    """
    Simula o sistema incremental com sequência de delta_u
    
    Args:
        model: instância da classe linDiscretizeComp
        delta_u_sequence: sequência de variações de entrada (delta_u)
    
    Returns:
        X_hist: histórico dos estados
        Y_hist: histórico das saídas
        U_hist: histórico das entradas (valores absolutos)
    """
    # Sistema incremental
    Atil, Btil, Ctil, Dtil, A, B, C, D = model.SysD
    
    N_sim = delta_u_sequence.shape[1]
    
    # Inicialização: estado no ponto de operação
    x_k = np.zeros((model.nX, 1))  # Normalizado: zero no ponto de operação
    u_k = np.zeros((model.nU, 1))  # Normalizado: zero no ponto de operação
    
    # Estado incremental (estado aumentado)
    xi_k = np.vstack([x_k, u_k])  # [x_k; u_{k-1}]
    
    # Históricos
    X_hist = np.zeros((model.nX, N_sim))
    Y_hist = np.zeros((model.nY, N_sim))
    U_hist = np.zeros((model.nU, N_sim))
    
    for k in range(N_sim):
        # Variação da entrada (delta_u) - já deve estar normalizada
        delta_u = delta_u_sequence[:, k:k+1]
        
        # Atualização do estado incremental
        xi_k = Atil @ xi_k + Btil @ delta_u
        
        # Extrair estados e entrada atual
        x_k = xi_k[:model.nX]
        u_k = xi_k[model.nX:]
        
        # Saída
        y_k = Ctil @ xi_k + Dtil @ delta_u
        
        # Armazenar (valores reais, não normalizados)
        X_hist[:, k] = model.denormalize(x_k, 'x').flatten()
        Y_hist[:, k] = model.denormalize(y_k, 'y').flatten()
        U_hist[:, k] = model.denormalize(u_k, 'u').flatten()
    
    return X_hist, Y_hist, U_hist

# FUNÇÃO PARA SIMULAR PLANTA
def simulate_plant(model, u_sequence):
    """
    Simula a planta fenomenológica com sequência de entradas absolutas
    
    Args:
        model: instância da classe linDiscretizeComp
        u_sequence: sequência de entradas absolutas
    
    Returns:
        plant_hist_y: histórico das saídas da planta
    """
    N_sim = u_sequence.shape[1]
    
    # Condições iniciais da planta
    x_plant = model.x_ss.flatten()
    z_plant = model.z_ss.flatten()
    
    # Histórico
    plant_hist_y = np.zeros((model.nY, N_sim))
    
    # Criar simulador
    sim = SimuladorDuto(model.meu_sistema, dt=model.dt, n_steps=1)
    
    for k in range(N_sim):
        print(k)
        # Entrada atual (absoluta)
        u_current = u_sequence[:, k]
        
        # Executar um passo da simulação
        resultados = sim.run(x_plant, z_plant, u_current)
        
        # Atualizar estados
        x_plant[0::3] = resultados["T_sol"]
        x_plant[1::3] = resultados["V_sol"]
        x_plant[2::3] = resultados["w_sol"]
        z_plant = resultados["z_sol"].T
        
        # Extrair as saídas de interesse (y0 a y3)
        y_plant = np.array([
            z_plant[-6],  # y0
            z_plant[-3],  # y1  
            z_plant[-2],  # y2
            z_plant[-1]   # y3
        ])
        
        # Armazenar
        plant_hist_y[:, k] = y_plant.flatten()
    
    return plant_hist_y

# AVALIAÇÃO DO ESPAÇO DE ESTADOS
if __name__ == "__main__": 
    # --- Passo 1: Obtenção do sistema discretizado ---
    model = linDiscretizeComp()
    Atil, Btil, Ctil, Dtil, A, B, C, D = model.SysD
    
    print("Sistema carregado:")
    print(f"  Dimensão de Atil: {Atil.shape}")
    print(f"  Dimensão de Btil: {Btil.shape}")
    print(f"  Dimensão de Ctil: {Ctil.shape}")
    
    # --- Passo 2: Configuração da Simulação ---
    N_sim = 30  # 20 horas em passos de 10 minutos
    time_vec = np.arange(N_sim)
    time_vec_hours = time_vec * model.dt / 3600
    
    # Criar sequência de delta_u (variações de entrada normalizadas)
    delta_u_sequence = np.zeros((model.nU, N_sim))
    
    # Apenas 3 pequenos degraus
    # Degrau 1: +2% no tempo 50
    degrau1_idx = 15
    delta_u_sequence[0, degrau1_idx] = 0.02  # +2% normalizado
    
    # # Degrau 2: -3% no tempo 200
    # degrau2_idx = 60
    # delta_u_sequence[0, degrau2_idx] = -0.03  # -3% normalizado
    
    # # Degrau 3: +1.5% no tempo 350
    # degrau3_idx = 90
    # delta_u_sequence[0, degrau3_idx] = 0.015  # +1.5% normalizado
    
    print(f"\nDegraus aplicados:")
    print(f"  Tempo {degrau1_idx} ({degrau1_idx*model.dt/3600:.1f}h): +2%")
    # print(f"  Tempo {degrau2_idx} ({degrau2_idx*model.dt/3600:.1f}h): -3%")
    # print(f"  Tempo {degrau3_idx} ({degrau3_idx*model.dt/3600:.1f}h): +1.5%")
    
    # --- Passo 3: Simular Sistema Incremental ---
    print("\nSimulando sistema incremental...")
    X_ss_hist, Y_ss_hist, U_ss_hist = simulate_incremental_system(model, delta_u_sequence)
    
    # --- Passo 4: Calcular sequência de entradas absolutas para a planta ---
    # A sequência de entradas absolutas é U_ss_hist (já calculada pelo sistema incremental)
    print("\nSimulando planta fenomenológica...")
    plant_hist_y = simulate_plant(model, U_ss_hist)
    
    # --- Passo 5: Calcular métricas ---
    output_names = model.get_output_names()
    mse_outputs = np.zeros(model.nY)
    mae_outputs = np.zeros(model.nY)
    
    for i in range(model.nY):
        mse_outputs[i] = np.mean((Y_ss_hist[i, :] - plant_hist_y[i, :])**2)
        mae_outputs[i] = np.mean(np.abs(Y_ss_hist[i, :] - plant_hist_y[i, :]))
    
    # --- Passo 6: Plotar resultados ---
    destino_imagens = "C:/Users/guilh/OneDrive/Documents/Workspace/UFBA/ENGG21 - Controle Avançado/EMPC/graficos/"
    os.makedirs(destino_imagens, exist_ok=True)
    
    # 6.1. Figura 1: Entradas e Saídas
    fig1, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Subplot 1: Entrada (velocidade do compressor)
    ax1 = axes[0]
    ax1.plot(time_vec_hours, U_ss_hist[0, :], 'b-', linewidth=2)
    ax1.axvline(x=degrau1_idx*model.dt/3600, color='r', linestyle='--', alpha=0.5)
    # ax1.axvline(x=degrau2_idx*model.dt/3600, color='r', linestyle='--', alpha=0.5)
    # ax1.axvline(x=degrau3_idx*model.dt/3600, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Velocidade / RPM')
    ax1.set_title('Entrada: Velocidade do Compressor')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Saídas y0 e y1
    ax2 = axes[1]
    ax2.plot(time_vec_hours, Y_ss_hist[0, :], 'b-', linewidth=2, label='ES - y0')
    ax2.plot(time_vec_hours, plant_hist_y[0, :], 'r--', linewidth=2, label='Planta - y0')
    ax2.plot(time_vec_hours, Y_ss_hist[1, :], 'g-', linewidth=2, label='ES - y1')
    ax2.plot(time_vec_hours, plant_hist_y[1, :], 'm--', linewidth=2, label='Planta - y1')
    ax2.set_ylabel('Saídas y0, y1')
    ax2.set_title(f'Saídas y0 (MSE: {mse_outputs[0]:.2e}) e y1 (MSE: {mse_outputs[1]:.2e})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Subplot 3: Saídas y2 e y3
    ax3 = axes[2]
    ax3.plot(time_vec_hours, Y_ss_hist[2, :], 'b-', linewidth=2, label='ES - y2')
    ax3.plot(time_vec_hours, plant_hist_y[2, :], 'r--', linewidth=2, label='Planta - y2')
    ax3.plot(time_vec_hours, Y_ss_hist[3, :], 'g-', linewidth=2, label='ES - y3')
    ax3.plot(time_vec_hours, plant_hist_y[3, :], 'm--', linewidth=2, label='Planta - y3')
    ax3.set_xlabel('Tempo / h')
    ax3.set_ylabel('Saídas y2, y3')
    ax3.set_title(f'Saídas y2 (MSE: {mse_outputs[2]:.2e}) e y3 (MSE: {mse_outputs[3]:.2e})')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    plt.suptitle('Comparação: Sistema Incremental vs Planta Fenomenológica', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(destino_imagens, 'comparacao_completa.png'), dpi=300)
    plt.savefig(os.path.join(destino_imagens, 'comparacao_completa.pdf'))
    plt.show()
    
    # 6.2. Figura 2: Erros individuais
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(model.nY):
        ax = axes[i]
        erro = Y_ss_hist[i, :] - plant_hist_y[i, :]
        ax.plot(time_vec_hours, erro, 'k-', linewidth=1.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.fill_between(time_vec_hours, 0, erro, where=erro>0, alpha=0.3, color='blue', label='ES > Planta')
        ax.fill_between(time_vec_hours, 0, erro, where=erro<0, alpha=0.3, color='red', label='ES < Planta')
        ax.set_xlabel('Tempo / h')
        ax.set_ylabel(f'Erro {output_names[i]}')
        ax.set_title(f'{output_names[i]} - Erro (MSE: {mse_outputs[i]:.2e}, MAE: {mae_outputs[i]:.4f})')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.suptitle('Erros entre Espaço de Estados Incremental e Planta', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(destino_imagens, 'erros_individuais.png'), dpi=300)
    plt.savefig(os.path.join(destino_imagens, 'erros_individuais.pdf'))
    plt.show()
    
    # 6.3. Figura 3: Resumo estatístico
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras para métricas
    x_pos = np.arange(model.nY)
    width = 0.35
    
    ax3a.bar(x_pos - width/2, mse_outputs, width, label='MSE', color='skyblue')
    ax3a.bar(x_pos + width/2, mae_outputs, width, label='MAE', color='lightcoral')
    ax3a.set_xlabel('Saídas')
    ax3a.set_ylabel('Valor do Erro')
    ax3a.set_title('Métricas de Erro por Saída')
    ax3a.set_xticks(x_pos)
    ax3a.set_xticklabels(output_names)
    ax3a.legend()
    ax3a.grid(True, alpha=0.3, axis='y')
    
    # Texto com resumo
    summary_text = f"""RESUMO DA SIMULAÇÃO
Tempo total: {N_sim*model.dt/3600:.1f} horas
Passos: {N_sim}
dt: {model.dt} s ({model.dt/60:.1f} min)

Degraus aplicados:
1. +2% em {degrau1_idx*model.dt/3600:.1f}h

Erro médio:
MSE: {np.mean(mse_outputs):.2e}
MAE: {np.mean(mae_outputs):.4f}"""
    
    ax3b.text(0.05, 0.95, summary_text, transform=ax3b.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3b.axis('off')
    
    plt.suptitle('Análise Estatística - Sistema Incremental', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(destino_imagens, 'resumo_estatistico.png'), dpi=300)
    plt.savefig(os.path.join(destino_imagens, 'resumo_estatistico.pdf'))
    plt.show()
    
    # --- Passo 7: Imprimir resumo detalhado ---
    print("\n" + "="*80)
    print("RESUMO DETALHADO DA SIMULAÇÃO")
    print("="*80)
    print(f"Duração: {N_sim} passos = {N_sim*model.dt/3600:.1f} horas")
    print(f"Tempo de amostragem: {model.dt} s")
    print(f"Entrada nominal (u_ss): {model.u_ss.flatten()[0]:.1f} RPM")
    
    print("\n" + "-"*80)
    print("DEGRAUS APLICADOS (delta_u normalizado):")
    print("-"*80)
    print(f"  t = {degrau1_idx} ({degrau1_idx*model.dt/3600:.1f}h): +0.02")
    # print(f"  t = {degrau2_idx} ({degrau2_idx*model.dt/3600:.1f}h): -0.03")
    # print(f"  t = {degrau3_idx} ({degrau3_idx*model.dt/3600:.1f}h): +0.015")
    