import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from statsmodels.graphics.tsaplots import plot_acf, plot_ccf
import os
import time 

imagesPath = os.path.join(os.getcwd(), 'MPC-Freire/plots')
os.makedirs(imagesPath, exist_ok=True)

# Constantes Fundamentais
g = 9.81  # Aceleração da gravidade [m/s²]
R = 8.314 # Constante dos gases [J/mol.K]

# Parâmetros Gerais do Sistema 
M = 0.020 # Massa molar do gás [kg/mol]
ro_o = 800 # Densidade do óleo no reservatório [kg/m³] 
Ps = 2e6  # Pressão do separador [Pa]
vo = 1 / ro_o # Volume específico do óleo [m³/kg]
GOR_geral = 0.1 # Razão Gás-Óleo [kg/kg] 
# Parâmetros do Riser
Dr = 0.121 # Diâmetro do riser [m]
Hr_riser_comum = 500 # Altura vertical do riser
Tr_riser_comum = 301 # Temperatura do riser [K] 
Crh = 10e-3 # Coeficiente da válvula da cabeça do riser [m²]
Ar_riser_comum = (ca.pi * (Dr ** 2)) / 4 # Área da seção transversal do riser [m²]
Ta = 28 + 273.15

Lr_comum = 500 # Length do riser comum
Ar_comum = 0.0114 # Area do riser comum
Crh = 10E-3
Tr_comum = 30 + 273.15 # Temperatura média no riser em K

dp_fric_t1 = 0
dp_fric_bh1 = 0
dp_fric_t2 = 0
dp_fric_bh2 = 0
dp_fric_r_comum = 0

class RiserModel:
    def __init__(self, p, m, steps, nY, nX, nU, dt, poco=1):
        # Contantes iniciais
        self.p = p
        self.m = m
        self.steps = steps
        self.nY = nY
        self.nX = nX
        self.nU = nU
        self.dt = dt
        self.y = []
        self.x = []
        self.u = []

        # Injeção inicial para simulação do sistema
        self.injInit = 3
        self.u0 = [self.injInit]*self.m

        # Geometria e Temperaturas
        self.Lw = 1500 # Comprimento do tubo [m]
        self.Dw = 0.121 # Diâmetro do poço [m]
        self.Hw = 1000 # Altura de coluna hidrostática no poço [m]
        self.Hbh = 500
        self.Lbh = 500 # Comprimento do poço abaixo do ponto de injeção [m]
        self.Dbh = 0.121 # Diâmetro da seção abaixo do ponto injeção [m]
        self.Tw = 305 # Temperatura no tubo [K]
        self.Lr = 500 # Comprimento do riser [m]
        self.La = 1500 # Comprimento da região anular [m]
        self.Dr = 0.121
        self.Da = 0.189 # Diâmetro do anular [m]
        self.Ha = 1000 # Altura da região anular [m]
        self.Ta = 301 # Temperatura da região anular [K]

        # Coeficientes e Propriedades
        self.PI = 0.7e-5 # Índice de Produtividade [kg/(s·Pa)]
        self.Cpc = 2e-3 # Coeficiente da choke de produção [m²]
        self.Civ = 0.1e-3 # Coeficiente da válvula de injeção [m²]
        self.Pr = 1.50e7 # Pressão no reservatório [Pa]
        self.Lr_poco = 500 # Distância do reservatório até o ponto de injeção 

        # Áreas e Volumes Calculados para Poço 1
        self.Aw = (ca.pi * (self.Dw ** 2)) / 4 # Área da seção transversal do poço [m²]
        self.Aa = (ca.pi * (self.Da ** 2)) / 4 - (ca.pi * (self.Dw ** 2)) / 4 # Área da seção transversal do anel [m²]
        self.Abh = (ca.pi * (self.Dbh ** 2)) / 4 # Área da seção transversal abaixo do ponto de injeção [m²]
        self.Ar = (ca.pi * (self.Dr ** 2)) / 4 
        self.Va = self.La * self.Aa # Volume da região anular [m³]

        if poco == 1:
            # --- Parâmetros do Poço 1 ---
            self.GOR = 0.08 # Razão Gás-Óleo [kg/kg] 39
        elif poco == 2:
            # --- Parâmetros do Poço 2 ---
            self.GOR = 0.1 # Razão Gás-Óleo [kg/kg] 39

        self.f_modelo = self.createModelo()
        self.caPredFun = self.caPredFunction()
    
    def fun(self, x, par):
        # x[0]: m_ga1 (massa de gás no anular do poço) 0
        # x[1]: m_gt1 (massa de gás no tubo do poço) 1
        # x[2]: m_ot1 (massa de óleo no tubo do poço) 2
        # x[3]: m_gr (massa de gás no riser comum) 3
        # x[4]: m_or (massa de óleo no riser comum) 4

        # Parâmetros: (Vazões de injeção de gás para cada poço)
        wgl = par[0]

        Pai = ((R * self.Ta / (self.Va * M)) + ((g * self.La) / (self.La * self.Aa))) * x[0]
        ro_w = (x[1] + x[2] - (ro_o * self.Lr * self.Ar)) / (self.Lw * self.Aw) 
        Pwh = (R * self.Tw / M) * (x[2] / (self.Lw * self.Aw + self.Lr * self.Ar - (vo * x[2]))) - 1/2 * ((x[1] + x[2])* g* self.Hw / (self.Lw * self.Aw)) 

        Pwi = Pwh + (g / (self.Lw * self.Aw)) * (x[2] + x[1] - ro_o * self.Lr * self.Ar) * self.Hw + dp_fric_t1 
        Pbh = Pwi + (ro_w * g * self.Hbh) + dp_fric_bh1
        ro_a = (M * Pai) / (R * Ta)

        wiv = self.Civ * ca.sqrt(ca.fmax(0, ro_a * (Pai - Pwi)))
        wro = self.PI * (self.Pr - Pbh)
        wrg = self.GOR * wro

        ro_r = (x[3] + x[4]) / (Lr_comum * Ar_comum)
        Prh = (Tr_comum * R / M) * (x[3] / (self.Lbh * self.Abh)) 
        Pm = Prh + (ro_r * g * self.Hbh) + dp_fric_r_comum 

        # Vazões de produção na cabeça do poço (para cada poço, dependendo do Pm comum)
        y3 = ca.fmax(0, (Pwh - Pm))
        wpc = self.Cpc * ca.sqrt(ro_w * y3) # Sua fórmula original de wpc
        wpg_prod = (x[1] / (x[1] + x[2])) * wpc
        wpo_prod = (x[1] / (x[1] + x[2])) * wpc

        # Vazões totais no riser
        y4_rh = ca.fmax(0, (Prh - Ps))
        wrh = Crh * ca.sqrt(ro_r * y4_rh)
        wtg = (x[3] / (x[3] + x[4])) * wrh
        wto = (x[3] / (x[3] + x[4])) * wrh

        # --- Derivadas dos Estados ---
        dx0 = wgl - wiv  # m_ga1
        dx1 = wiv + wrg - wpg_prod  # m_gt1 (usando wpg1_prod)
        dx2 = wro - wpo_prod  # m_ot1 (usando wpo1_prod)
        dx3 = wpg_prod - wtg  # m_gr (somando contribuições)
        dx4 = wpo_prod - wto  # m_or (somando contribuições)

        ode = ca.vertcat(dx0, dx1, dx2, dx3, dx4, wpg_prod, wpo_prod)

        return ode
    
    def createModelo(self):
        # Modelo de nicole
        '''Você pode colocar qualquer modelo de qualquer sistema aqui,
          pois é um MPC não linear, MAS você precisa fazer as alterações necessárias
          ou seja, quantidade de variáveis, como o modelo vai retornar,
          se isso altera algo no NMPC (caso você esteja trabalhando com
          estados = saídas, você deve remanejar muitas coisas)'''
        
        x = ca.MX.sym('x', 7)  # 8 estados do sistema
        par = ca.MX.sym('par', 3)  # 3 parâmetros
        t = ca.MX.sym('t')

        # Integração
        rhs = self.fun(x, par)

        dae = {'x': x, 'p': par, 'ode': rhs}
        opts = {'tf': self.dt}
        integrator = ca.integrator('integrator', 'rk', dae, opts)

        res = integrator(x0=x, p=par)
        x_new = res['xf']

        x_new[3] = x_new[3] + x[5] # x[5] wpg_prod do outro poço
        x_new[4] = x_new[4] + x[6] # x[6] wpo_prod do outro poço

        wgl = par[0]

        Pai = ((R * self.Ta / (self.Va * M)) + ((g * self.La) / (self.La * self.Aa))) * x[0]
        ro_w = (x[1] + x[2] - (ro_o * self.Lr * self.Ar)) / (self.Lw * self.Aw) 
        Pwh = (R * self.Tw / M) * (x[2] / (self.Lw * self.Aw + self.Lr * self.Ar - (vo * x[2]))) - 1/2 * ((x[1] + x[2])* g* self.Hw / (self.Lw * self.Aw)) 

        Pwi = Pwh + (g / (self.Lw * self.Aw)) * (x[2] + x[1] - ro_o * self.Lr * self.Ar) * self.Hw + dp_fric_t1 
        Pbh = Pwi + (ro_w * g * self.Hbh) + dp_fric_bh1
        ro_a = (M * Pai) / (R * Ta)
        wiv = self.Civ * ca.sqrt(ca.fmax(0, ro_a * (Pai - Pwi)))
        wro = self.PI * (self.Pr - Pbh)
        wrg = self.GOR * wro

        # --- Cálculos para o Riser Comum ---
        ro_r = (x[3] + x[4]) / (Lr_comum * Ar_comum)
        Prh = (Tr_comum * R / M) * (x[3] / (self.Lbh * self.Abh))
        Pm = Prh + (ro_r * g * self.Hbh) + dp_fric_r_comum

        y3 = ca.fmax(0, (Pwh - Pm))
        wpc = self.Cpc * ca.sqrt(ro_w * y3)
        wpg_prod = (x[1] / (x[1] + x[2])) * wpc
        wpo_prod = (x[1] / (x[1] + x[2])) * wpc

        y4_rh = ca.fmax(0, (Prh - Ps))
        wrh = Crh * ca.sqrt(ro_r * y4_rh)
        wtg = (x[3] / (x[3] + x[4])) * wrh
        wto = (x[3] / (x[3] + x[4])) * wrh

        # outputs = ca.vertcat(
        #     wiv1, wro1, wpc1, wpg1_prod, wpo1_prod, Pai1, Pwh1, Pwi1, Pbh1, wrg1, # Poço 1
        #     wiv2, wro2, wpc2, wpg2_prod, wpo2_prod, Pai2, Pwh2, Pwi2, Pbh2, wrg2, # Poço 2
        #     wtg, wto, Prh, Pm, ro_r, wrh # Riser e variáveis comuns
        # )

        # Aqui meu sistema só está retornando as pressões de fundo de poço
        # Você pode alterar isso se quiser, eu diminui, pois só estava analisando essas
        # E precisava diminuir o tempo de cálculo das Hessianas
        outputs = ca.vertcat(Pbh, wpo_prod)

        return ca.Function('f_modelo', [x, par], [outputs, x_new], ['x', 'par'], ['outputs', 'x_new'])
    
    def caPredFunction(self):
        # Modelo utilizado pelo controlador MPC
        # Retorna uma função casadi que retorna o trend já no tamanho de P pontos
        x0 = ca.MX.sym('x0', self.nX*self.m, 1)  # Vetor de estados iniciais
        u0 = ca.MX.sym('u0', self.nU*self.m, 1)  # Vetor de parâmetros de controle iniciais
        dU = ca.MX.sym('dU', self.nU*self.m, 1)  # Variação dos parâmetros de controle

        init_x = x0[-self.nX:]  # Últimos estados iniciais

        y = ca.MX()
        x = ca.MX()
        u = ca.MX()
        u0_curr = u0

        for j in range(self.p):
            if j < self.m:
                u0_curr = ca.vertcat(u0_curr, u0_curr[-self.nU] + dU[self.nU*j])
                u0_curr = ca.vertcat(u0_curr, u0_curr[-self.nU])
                u0_curr = ca.vertcat(u0_curr, u0_curr[-self.nU])
                u0_curr = u0_curr[self.nU:]
            par = ca.vertcat(u0_curr[-self.nU], u0_curr[-self.nU+1], u0_curr[-self.nU+2])
            outputs, init_x = self.f_modelo(init_x, par)
            y = ca.vertcat(y, outputs)
            x = ca.vertcat(x, init_x)
            if j < self.m:
                u = ca.vertcat(u, par)

        out_trend = ca.Function('y_trend', [x0, u0, dU], [y, x, u])

        return out_trend
    
    def pPlanta(self, x0, dU):
        # Modelo da planta 
        # é importante colocar uma perturbação na planta 
        # para evitar ser um controlador nominal
        self.x = []
        self.y = []
        init_x = x0[-self.nX:]  # Últimos estados iniciais

        self.u0.append(self.u0[-self.nU] + dU[0])

        par = np.array([self.u0[-self.nU]])
        outputs, init_x = self.f_modelo(init_x, par)
        # stds = np.array([5e3, 5e3, 0.01, 0.01])  # desvio padrão para cada saída
        # noise = np.random.normal(0, stds.reshape(-1, 1))
        # outputs += noise

        self.y.append(outputs.full().flatten())
        self.x.append(init_x.full().flatten())
        self.u.append([self.u0[-self.nU]])
            
        self.y = np.array(self.y).reshape(-1,1)
        self.x = np.array(self.x).reshape(-1,1)
        self.uk = np.array(self.u).reshape(-1,1)[-self.nU:]
        return self.y, self.x, self.uk
    
    def pIniciais(self):
        # Pontos iniciais para começar a simulação
        y = []
        x = []
        u = []

        u0 = [self.injInit]*self.nU
        x0 = np.array([1961.8804936, 2017.33517325, 962.04921361, 1042.48337861,
                           6939.02402957, 6617.93163452, 117.97660766, 795.94318092])  # Estados iniciais

        t0 = 1
        tf = 4000
        dt = self.dt
        t = np.arange(t0, tf, dt)

        for ti in t:
            y0, x0 = self.f_modelo(x0, u0[-2:])
            y.append(y0.full().flatten())
            x.append(x0.full().flatten())
            u.append(u0[-2:])

        y0 = np.array(y[-self.steps:]).reshape(-1,1).reshape(-1,1) # [-self.steps:][-self.steps:]).reshape(-1,1).reshape(-1,1)
        x0 = np.array(x[-self.steps:]).reshape(-1,1).reshape(-1,1)#
        u0 = np.array(u[-self.steps:]).reshape(-1,1).reshape(-1,1)#

        return y0, x0, u0
    
    def setPoints(self, nSP):
        # Setpoints para o controle, não estou usando no código atualmente
        # Defina os setpoints manualmente no nmpc.py
        def fun_wrap(x, par1, par2):
            x_casadi = ca.DM(x)
            result_casadi = self.fun(x_casadi, [par1, par2])
            return np.array(result_casadi).flatten()

        SPlist = []
        for i in range(nSP):
            par1 = np.random.randint(0, 10)/5
            par2 = np.random.randint(0, 10)/5
            result = fsolve(fun_wrap, (1961.8804936, 2017.33517325, 962.04921361, 1042.48337861,
                           6939.02402957, 6617.93163452, 117.97660766, 795.94318092), args=(par1, par2))
            x_eq = ca.DM(result)
            y_eq = self.f_modelo(x_eq, [par1, par2])[0]
            SPlist.append([y_eq, x_eq, [par1, par2]])

        return SPlist

if __name__ == "__main__":
    poco1 = RiserModel(p=400, m=400, steps=3, dt=5, nY=3, nX=7, nU=1, poco=1)
    poco2 = RiserModel(p=400, m=400, steps=3, dt=5, nY=3, nX=7, nU=1, poco=2)
    
    # Parâmetros iniciais
    x_poco1 = np.array([1961.8804936, 962.04921361,
                           6939.02402957, 117.97660766, 795.94318092, 30, 30]).reshape(-1, 1)
    x_poco2 = np.array([2017.33517325, 1042.48337861,
                           6617.93163452, 117.97660766, 795.94318092, 30, 30]).reshape(-1, 1)
    
    N = 400
    
    y_poco1 = poco1.u0.copy()
    y_poco2 = poco2.u0.copy()

    y_hist1 = []
    x_hist1 = []
    u_hist1 = []
    y_hist2 = []
    x_hist2 = []
    u_hist2 = []

    for i in range(N):
        # Simulação do Poço 1
        y_poco1, x_poco1, u_poco1 = poco1.pPlanta(x_poco1, [0])
        y_hist1.append(y_poco1)
        x_hist1.append(x_poco1)
        u_hist1.append(u_poco1)
        x_poco2[5:] = x_poco1[5:]  # Atualiza os estados do Poço 2 com os do Poço 1
        # Simulação do Poço 2
        y_poco2, x_poco2, u_poco2 = poco2.pPlanta(x_poco2, [0])
        y_hist2.append(y_poco2)
        x_hist2.append(x_poco2)
        u_hist2.append(u_poco2)
        x_poco1[5:] = x_poco2[5:]  # Atualiza os estados do Poço 1 com os do Poço 2
    
    y_hist1 = np.array(y_hist1).reshape(-1, 1)
    x_hist1 = np.array(x_hist1).reshape(-1, 1)
    u_hist1 = np.array(u_hist1).reshape(-1, 1)
    y_hist2 = np.array(y_hist2).reshape(-1, 1)
    x_hist2 = np.array(x_hist2).reshape(-1, 1)
    u_hist2 = np.array(u_hist2).reshape(-1, 1)

    # === PLOT ===
    t = np.arange(800)

    print(f"{y_hist1[:,0][-1]} - Pressão")
    print(f"{y_hist1[:,1][-1]} - Vazao de oleo")
    print(f"{y_hist1[:,2][-1]} - Vazao de gas")

    plt.figure(figsize=(16, 14))

    plt.subplot(3, 1, 1)
    # plt.plot(t, u_hist[0][::2], label="wgl1 (Poço 1)")
    # plt.plot(t, u_hist[0][1::2], label="wgl2 (Poço 2)")
    plt.plot(t, umk[:,0], label="wgl1 (P inicais)")
    plt.plot(t, umk[:,1], label="wgl2 (P inicais)")
    plt.ylabel("Injeção de gás [kg/s]")
    plt.title("Controles aplicados")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    # plt.plot(t, y_out[:, 2], label="Óleo produzido Poço 1")
    # plt.plot(t, y_out[:, 3], label="Óleo produzido Poço 2")
    plt.plot(t, ymk[:,2], label="wgl1 (P inicais)")
    plt.plot(t, ymk[:,3], label="wgl2 (P inicais)")
    plt.ylabel("Vazão [kg/s]")
    plt.title("Vazão de Óleo Produzido")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    # plt.plot(t, y_out[:, 0], label="Pbh Poço 1 (Pressão de Fundo)")
    # plt.plot(t, y_out[:, 1], label="Pbh Poço 2 (Pressão de Fundo)")
    plt.plot(t, ymk[:,0], label="wgl1 (P inicais)")
    plt.plot(t, ymk[:,1], label="wgl2 (P inicais)")
    plt.ylabel("Pressão [Pa]")
    plt.xlabel("Tempo [s]")
    plt.title("Pressão de Fundo dos Poços")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "sim.png"))
    
    n_lags = 99

    # Plot Auto-correlation Function (ACF) for inputs and outputs
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plot_acf(ymk[:100, 0], lags=n_lags, title='ACF: Pbh1 (Output 1)', ax=plt.gca())
    plt.xlabel(f'Lags (seconds, dt={5})')
    plt.ylabel('Autocorrelation')

    plt.subplot(2, 2, 2)
    plot_acf(ymk[:100, 1], lags=n_lags, title='ACF: Pbh2 (Output 2)', ax=plt.gca())
    plt.xlabel(f'Lags (seconds, dt={5})')
    plt.ylabel('Autocorrelation')

    plt.subplot(2, 2, 3)
    plot_acf(ymk[:100, 2], lags=n_lags, title='ACF: wpo1_prod (Output 3)', ax=plt.gca())
    plt.xlabel(f'Lags (seconds, dt={5})')
    plt.ylabel('Autocorrelation')

    plt.subplot(2, 2, 4)
    plot_acf(ymk[:100, 3], lags=n_lags, title='ACF: wpo2_prod (Output 4)', ax=plt.gca())
    plt.xlabel(f'Lags (seconds, dt={5})')
    plt.ylabel('Autocorrelation')

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "acf_plots.png"))
    plt.show()

