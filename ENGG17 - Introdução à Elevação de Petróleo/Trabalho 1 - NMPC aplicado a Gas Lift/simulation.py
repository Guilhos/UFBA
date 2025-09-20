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
# --- Parâmetros do Poço 1 ---
# Geometria e Temperaturas
Lw1 = 1500 # Comprimento do tubo [m]
Dw1 = 0.121 # Diâmetro do poço [m]
Hw1 = 1000 # Altura de coluna hidrostática no poço [m]
Hbh1 = 500
Lbh1 = 500 # Comprimento do poço abaixo do ponto de injeção [m]
Dbh1 = 0.121 # Diâmetro da seção abaixo do ponto injeção [m]
Tw1 = 305 # Temperatura no tubo [K]

La1 = 1500 # Comprimento da região anular [m]
Dr = 0.121
Da1 = 0.189 # Diâmetro do anular [m]
Ha1 = 1000 # Altura da região anular [m]
Ta1 = 301 # Temperatura da região anular [K]

# Coeficientes e Propriedades
PI1 = 0.7e-5 # Índice de Produtividade [kg/(s·Pa)]
Cpc1 = 2e-3 # Coeficiente da choke de produção [m²]
Civ1 = 0.1e-3 # Coeficiente da válvula de injeção [m²]
Lr_poco1 = 500 # Distância do reservatório até o ponto de injeção 
GOR1 = 0.08 # Razão Gás-Óleo [kg/kg] 39

# Áreas e Volumes Calculados para Poço 1
Aw1 = (ca.pi * (Dw1 ** 2)) / 4 # Área da seção transversal do poço [m²]
Aa1 = (ca.pi * (Da1 ** 2)) / 4 - (ca.pi * (Dw1 ** 2)) / 4 # Área da seção transversal do anel [m²]
Abh1 = (ca.pi * (Dbh1 ** 2)) / 4 # Área da seção transversal abaixo do ponto de injeção [m²]
Ar1 = (ca.pi * (Dr ** 2)) / 4 
Va1 = La1 * Aa1 # Volume da região anular [m³]

# --- Parâmetros do Poço 2 
# Geometria e Temperaturas
Lw2 = 1500
Lr2 = 500
Dw2 = 0.121
Hw2 = 1000
Lbh2 = 500
Hbh2 = 500
Dbh2 = 0.121
Tw2 = 305
Lr1 = 500
La2 = 1500
Da2 = 0.189
Dr= 0.121
Ha2 = 1000
Ta2 = 301

# Coeficientes e Propriedades
PI2 = 0.7e-5 # Pode ser diferente, exemplo: 0.23e-5
Cpc2 = 2e-3
Civ2 = 0.1e-3
Lr_poco2 = 500
GOR2 = 0.1 # Diferente do poço 1

# Áreas e Volumes Calculados para Poço 2
Aw2 = (ca.pi * (Dw2 ** 2)) / 4
Aa2 = (ca.pi * (Da2 ** 2)) / 4 - (ca.pi * (Dw2 ** 2)) / 4
Abh2 = (ca.pi * (Dbh2 ** 2)) / 4
Ar2 = (ca.pi * (Dr ** 2)) / 4 
Va2 = La2 * Aa2

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
    def __init__(self, p, m, steps, nY, nX, nU, dt):
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

        self.Pr1 = 1.50e7
        self.Pr2 = 1.55e7

        # Injeção inicial para simulação do sistema
        self.injInit = 3
        self.u0 = [self.injInit]*self.m*self.nU

        self.f_modelo = self.createModelo()
        self.caPredFun = self.caPredFunction()
    
    def fun(self, x, par, pr):
        # x[0]: m_ga1 (massa de gás no anular do poço 1)
        # x[1]: m_ga2 (massa de gás no anular do poço 2)
        # x[2]: m_gt1 (massa de gás no tubo do poço 1)
        # x[3]: m_gt2 (massa de gás no tubo do poço 2)
        # x[4]: m_ot1 (massa de óleo no tubo do poço 1)
        # x[5]: m_ot2 (massa de óleo no tubo do poço 2)
        # x[6]: m_gr (massa de gás no riser comum)
        # x[7]: m_or (massa de óleo no riser comum)

        # Parâmetros: (Vazões de injeção de gás para cada poço)
        wgl1, wgl2 = par[0], par[1]

        Pai1 = ((R * Ta1 / (Va1 * M)) + ((g * La1) / (La1 * Aa1))) * x[0]
        ro_w1 = (x[2] + x[4] - (ro_o * Lr1 * Ar1)) / (Lw1 * Aw1) 
        Pwh1 = (R * Tw1 / M) * (x[2] / (Lw1 * Aw1 + Lr1 * Ar1 - (vo * x[4]))) - 1/2 * ((x[2] + x[4])* g* Hw1 / (Lw1 * Aw1)) 

        Pwi1 = Pwh1 + (g / (Lw1 * Aw1)) * (x[4] + x[2] - ro_o * Lr1 * Ar1) * Hw1 + dp_fric_t1 
        Pbh1 = Pwi1 + (ro_w1 * g * Hbh1) + dp_fric_bh1
        ro_a1 = (M * Pai1) / (R * Ta)
        wiv1 = Civ1 * ca.sqrt(ca.fmax(0, ro_a1 * (Pai1 - Pwi1)))
        wro1 = PI1 * (pr[0] - Pbh1)
        wrg1 = GOR1 * wro1
    
        Pai2 = ((R * Ta2 / (Va2 * M)) + ((g * La2) / (La2 * Aa2))) * x[1]
        ro_w2 = (x[3] + x[5] - (ro_o * Lr2 * Ar2)) / (Lw2 * Aw2)
        Pwh2 = (R * Tw2 / M) * (x[3] / (Lw2 * Aw2 + Lr2 * Ar2 - (vo * x[5]))) - 1/2 * ((x[3] + x[5])* g* Hw2 / (Lw2 * Aw2))

        ro_a2 = (M * Pai2) / (R * Ta)
        Pwi2 = Pwh2 + (g / (Lw2 * Aw2)) * (x[5] + x[3] - ro_o * Lr2 * Ar2) * Hw2 + dp_fric_t2
        Pbh2 = Pwi2 + (ro_w2 * g * Hbh2) + dp_fric_bh2

        wiv2 = Civ2 * ca.sqrt(ca.fmax(0, ro_a2 * (Pai2 - Pwi2)))
        wro2 = PI2 * (pr[1] - Pbh2)
        wrg2 = GOR2 * wro2

        ro_r = (x[6] + x[7]) / (Lr_comum * Ar_comum)
        Prh = (Tr_comum * R / M) * (x[6] / (Lbh1 * Abh1)) 
        Pm = Prh + (ro_r * g * Hbh1) + dp_fric_r_comum 

        # Vazões de produção na cabeça do poço (para cada poço, dependendo do Pm comum)
        y3_1 = ca.fmax(0, (Pwh1 - Pm))
        wpc1 = Cpc1 * ca.sqrt(ro_w1 * y3_1) # Sua fórmula original de wpc
        wpg1_prod = (x[2] / (x[2] + x[4])) * wpc1
        wpo1_prod = (x[4] / (x[2] + x[4])) * wpc1

        y3_2 = ca.fmax(0, (Pwh2 - Pm))
        wpc2 = Cpc2 * ca.sqrt(ro_w2 * y3_2)
        wpg2_prod = (x[3] / (x[3] + x[5])) * wpc2
        wpo2_prod = (x[5] / (x[3] + x[5])) * wpc2

        # Vazões totais no riser
        y4_rh = ca.fmax(0, (Prh - Ps))
        wrh = Crh * ca.sqrt(ro_r * y4_rh)
        wtg = (x[6] / (x[6] + x[7])) * wrh
        wto = (x[7] / (x[6] + x[7])) * wrh

        # --- Derivadas dos Estados ---
        dx0 = wgl1 - wiv1  # m_ga1
        dx1 = wgl2 - wiv2  # m_ga2
        dx2 = wiv1 + wrg1 - wpg1_prod  # m_gt1 (usando wpg1_prod)
        dx3 = wiv2 + wrg2 - wpg2_prod  # m_gt2 (usando wpg2_prod)
        dx4 = wro1 - wpo1_prod  # m_ot1 (usando wpo1_prod)
        dx5 = wro2 - wpo2_prod  # m_ot2 (usando wpo2_prod)
        dx6 = wpg1_prod + wpg2_prod - wtg  # m_gr (somando contribuições)
        dx7 = wpo1_prod + wpo2_prod - wto  # m_or (somando contribuições)

        ode = ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)

        return ode
    
    def createModelo(self):
        # Modelo de nicole
        '''Você pode colocar qualquer modelo de qualquer sistema aqui,
          pois é um MPC não linear, MAS você precisa fazer as alterações necessárias
          ou seja, quantidade de variáveis, como o modelo vai retornar,
          se isso altera algo no NMPC (caso você esteja trabalhando com
          estados = saídas, você deve remanejar muitas coisas)'''
        
        x = ca.MX.sym('x', 8)  # 8 estados do sistema
        par = ca.MX.sym('par', 2)  # 2 parâmetros
        pr = ca.MX.sym('pr', 2)  # Parâmetro adicional, se necessário
        t = ca.MX.sym('t')

        p_all = ca.vertcat(par,pr)

        # Integração
        rhs = self.fun(x, par, pr)

        dae = {'x': x, 'p': p_all, 'ode': rhs}
        opts = {'tf': self.dt}
        integrator = ca.integrator('integrator', 'rk', dae, opts)

        res = integrator(x0=x, p=p_all)
        x_new = res['xf']

        wgl1, wgl2 = par[0], par[1]

        Pai1 = ((R * Ta1 / (Va1 * M)) + ((g * La1) / (La1 * Aa1))) * x[0]
        ro_w1 = (x[2] + x[4] - (ro_o * Lr1 * Ar1)) / (Lw1 * Aw1) 
        Pwh1 = (R * Tw1 / M) * (x[2] / (Lw1 * Aw1 + Lr1 * Ar1 - (vo * x[4]))) - 1/2 * ((x[2] + x[4])* g* Hw1 / (Lw1 * Aw1)) 

        Pwi1 = Pwh1 + (g / (Lw1 * Aw1)) * (x[4] + x[2] - ro_o * Lr1 * Ar1) * Hw1 + dp_fric_t1 
        Pbh1 = Pwi1 + (ro_w1 * g * Hbh1) + dp_fric_bh1
        ro_a1 = (M * Pai1) / (R * Ta)
        wiv1 = Civ1 * ca.sqrt(ca.fmax(0, ro_a1 * (Pai1 - Pwi1)))
        wro1 = PI1 * (pr[0] - Pbh1)
        wrg1 = GOR1 * wro1
    
        Pai2 = ((R * Ta2 / (Va2 * M)) + ((g * La2) / (La2 * Aa2))) * x[1]
        ro_w2 = (x[3] + x[5] - (ro_o * Lr2 * Ar2)) / (Lw2 * Aw2)
        Pwh2 = (R * Tw2 / M) * (x[3] / (Lw2 * Aw2 + Lr2 * Ar2 - (vo * x[5]))) - 1/2 * ((x[3] + x[5])* g* Hw2 / (Lw2 * Aw2))
        ro_a2 = (M * Pai2) / (R * Ta)

        Pwi2 = Pwh2 + (g / (Lw2 * Aw2)) * (x[5] + x[3] - ro_o * Lr2 * Ar2) * Hw2 + dp_fric_t2
        Pbh2 = Pwi2 + (ro_w2 * g * Hbh2) + dp_fric_bh2

        wiv2 = Civ2 * ca.sqrt(ca.fmax(0, ro_a2 * (Pai2 - Pwi2)))
        wro2 = PI2 * (pr[1] - Pbh2)
        wrg2 = GOR2 * wro2

        # --- Cálculos para o Riser Comum ---
        ro_r = (x[6] + x[7]) / (Lr_comum * Ar_comum)
        Prh = (Tr_comum * R / M) * (x[6] / (Lbh1 * Abh1))
        Pm = Prh + (ro_r * g * Hbh1) + dp_fric_r_comum

        y3_1 = ca.fmax(0, (Pwh1 - Pm))
        wpc1 = Cpc1 * ca.sqrt(ro_w1 * y3_1)
        wpg1_prod = (x[2] / (x[2] + x[4])) * wpc1
        wpo1_prod = (x[4] / (x[2] + x[4])) * wpc1

        y3_2 = ca.fmax(0, (Pwh2 - Pm))
        wpc2 = Cpc2 * ca.sqrt(ro_w2 * y3_2)
        wpg2_prod = (x[3] / (x[3] + x[5])) * wpc2
        wpo2_prod = (x[5] / (x[3] + x[5])) * wpc2

        y4_rh = ca.fmax(0, (Prh - Ps))
        wrh = Crh * ca.sqrt(ro_r * y4_rh)
        wtg = (x[6] / (x[6] + x[7])) * wrh
        wto = (x[7] / (x[6] + x[7])) * wrh

        # outputs = ca.vertcat(
        #     wiv1, wro1, wpc1, wpg1_prod, wpo1_prod, Pai1, Pwh1, Pwi1, Pbh1, wrg1, # Poço 1
        #     wiv2, wro2, wpc2, wpg2_prod, wpo2_prod, Pai2, Pwh2, Pwi2, Pbh2, wrg2, # Poço 2
        #     wtg, wto, Prh, Pm, ro_r, wrh # Riser e variáveis comuns
        # )

        # Aqui meu sistema só está retornando as pressões de fundo de poço
        # Você pode alterar isso se quiser, eu diminui, pois só estava analisando essas
        # E precisava diminuir o tempo de cálculo das Hessianas
        outputs = ca.vertcat(Pbh1, Pbh2, wpo1_prod, wpo2_prod)

        return ca.Function('f_modelo', [x, par, pr], [outputs, x_new], ['x', 'par', 'pr'], ['outputs', 'x_new'])
    
    def caPredFunction(self):
        # Modelo utilizado pelo controlador MPC
        # Retorna uma função casadi que retorna o trend já no tamanho de P pontos
        x0 = ca.MX.sym('x0', self.nX, 1)  # Vetor de estados iniciais
        u0 = ca.MX.sym('u0', self.nU, 1)  # Vetor de parâmetros de controle iniciais
        dU = ca.MX.sym('dU', self.nU*self.m, 1)  # Variação dos parâmetros de controle

        init_x = x0[-self.nX:]  # Últimos estados iniciais

        y = ca.MX()
        x = ca.MX()
        u = ca.MX()
        u0_curr = u0

        for j in range(self.p):
            if j < self.m:
                u0_curr = ca.vertcat(u0_curr, u0_curr[-self.nU] + dU[self.nU*j])
                u0_curr = ca.vertcat(u0_curr, u0_curr[-self.nU] + dU[self.nU*j+1])
                u0_curr = u0_curr[self.nU:]
            par = ca.vertcat(u0_curr[-self.nU], u0_curr[-self.nU+1])
            outputs, init_x = self.f_modelo(init_x, par, [self.Pr1, self.Pr2])
            y = ca.vertcat(y, outputs)
            x = ca.vertcat(x, init_x)
            if j < self.m:
                u = ca.vertcat(u, par)

        out_trend = ca.Function('y_trend', [x0, u0, dU], [y, x, u])

        return out_trend
    
    def pPlanta(self, x0, dU, iter):
        # Modelo da planta 
        # é importante colocar uma perturbação na planta 
        # para evitar ser um controlador nominal
        self.x = []
        self.y = []
        init_x = x0[-self.nX:]  # Últimos estados iniciais

        self.u0.append(self.u0[-self.nU] + dU[0])
        self.u0.append(self.u0[-self.nU] + dU[1])
        par = np.array([self.u0[-self.nU], self.u0[-self.nU+1]])
        # self.Pr1 = 1.50e7 + np.random.uniform(-0.1, 0.1) * 1e5
        # self.Pr2 = 1.55e7 + np.random.uniform(-0.1, 0.1) * 1e5

        outputs, init_x = self.f_modelo(init_x, par, [self.Pr1, self.Pr2])
        
        self.y.append(outputs.full().flatten())
        self.x.append(init_x.full().flatten())
        self.u.append([self.u0[-self.nU], self.u0[-self.nU+1]])
            
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
            y0, x0 = self.f_modelo(x0, u0[-2:], [self.Pr1, self.Pr2])
            y.append(y0.full().flatten())
            x.append(x0.full().flatten())
            u.append(u0[-2:])

        y0 = np.array(y[-1]).reshape(-1,1) # [-self.steps:][-self.steps:]).reshape(-1,1).reshape(-1,1)
        x0 = np.array(x[-1]).reshape(-1,1)#
        u0 = np.array(u[-1]).reshape(-1,1)#

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
    sim = RiserModel(p=400, m=400, steps=3, dt=5, nY=4, nX=8, nU=2)
    
    # Parâmetros iniciais
    N_sim = 4000  # número de iterações de simulação
    x_single = np.array([1961.8804936, 2017.33517325, 962.04921361, 1042.48337861,
                           6939.02402957, 6617.93163452, 117.97660766, 795.94318092])  # Estado único
    x0 = np.tile(x_single.reshape(-1, 1), (sim.m, 1))  # (8×1) empilhado 2 vezes → (16×1)
    dU_total = np.zeros((N_sim, sim.nU * sim.m, 1))  # Ex: 30 passos, cada com (4x1)
    # for k in range(N_sim):
    #     if k > 0 :
    #         dU_total[k][0] = dU_total[k][0] +0.25
    #         dU_total[k][1] = dU_total[k][1] +0.25
    #         dU_total[k][2] = dU_total[k][2] +0.25
    #         dU_total[k][3] = dU_total[k][3] +0.25
    #     if k == 11:
    #         dU_total[k][0] = dU_total[k][0] +0.25
    #         dU_total[k][1] = dU_total[k][1] +0.25
    #         dU_total[k][2] = dU_total[k][2] +0.25
    #         dU_total[k][3] = dU_total[k][3] +0.25
    
    dU_total[0][200] = 0.5
    dU_total[0][201] = -0.5
    
    dU_total[0][250] = 0.5
    
    dU_total[0][300] = 0.5
    
    dU_total[0][350] = 0.5
    
    dU_total[0][400] = 0.5
    dU_total[0][401] = -0.5
    
    dU_total[0][450] = 0.5
    
    dU_total[0][500] = 0.5
    
    dU_total[0][550] = 0.5
    
    dU_total[0][600] = 0.5
    
    dU_total[0][650] = 0.5
    
    u0 = np.ones((sim.nU * sim.m, 1)) * 3

    x_hist = []
    y_hist = []
    u_hist = []

    for k in range(1):
        dU = dU_total[k]

        # Obtem predição de saída e controle futuro (modelo interno)
        y_out, x_out, u_out = sim.caPredFun(x0, u0, dU)
        # y_pred = y_pred.full().flatten()
        # x_pred = x_pred.full().flatten()
        # u_pred = u_pred.full().flatten()
        u_out = np.array(u_out).reshape(-1,1)

        # Planta responde ao controle aplicado
        #y_out, x_out, u_out = sim.pPlanta(x0, dU)
        # Atualiza x0 empilhando os últimos m estados (mantém histórico)
        x0 = np.vstack((x0[sim.nX:], x_out[-sim.nX:]))  # Remove o mais antigo, adiciona novo

        # Atualiza histórico de controle
        u0 = np.vstack((u0[1:], [u_out[-2][0]]))
        u0 = np.vstack((u0[1:], [u_out[-1][0]]))

        x_hist.append(x_out[-sim.nY:])
        y_hist.append(y_out[-sim.nY:])
        u_hist.append(u_out.flatten())

    x_hist = np.array(x_hist)
    y_hist = np.array(y_hist)
    u_hist = np.array(u_hist)

    y_out = np.array(y_out).reshape(sim.p, sim.nY)

    ymk, xmk, umk = sim.pIniciais()

    print(ymk[-4:])
    print(y_hist[-1:])

    print(sim.setPoints(5))

    # === PLOT ===
    t = np.arange(800)

    print(f"{y_hist[:,0][-1]} - Pressão")
    print(f"{y_hist[:,2][-1]} - Vazao")

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

