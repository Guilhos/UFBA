import numpy as np
import control as ctrl 
import matplotlib.pyplot as plt
import cvxpy as cp
import model2ss
import os
import pickle

# --- CONFIGURAÇÃO GLOBAL DE FONTE ---
# Mantendo o tamanho global grande para eixos e títulos, 
# reduziremos manualmente na legenda dentro da função de plotagem.
plt.rcParams.update({'font.size': 35})
plt.rcParams.update({'axes.titlesize': 35})
plt.rcParams.update({'axes.labelsize': 35})
plt.rcParams.update({'xtick.labelsize': 35})
plt.rcParams.update({'ytick.labelsize': 35})
plt.rcParams.update({'legend.fontsize': 35}) 
plt.rcParams.update({'figure.titlesize': 35})

class EMPC:
    def __init__(self, model, p, m, Q, R, gamma):
        self.p = p
        self.m = m
        self.Q = Q
        self.R = R
        self.gamma = gamma
        self.model = model
        self.buildController()
        # Define o caminho para o arquivo de resultados
        self.results_path = os.path.join("ENGG21 - Controle Avançado\EMPC\libs", 'empc_results.pkl')

    def buildController(self):
        # Construção do MPC
        self.Atil, self.Btil, self.Ctil, self.Dtil, A, B, C, D = self.model.SysD

        # Matrizes Auxiliares:
        ## PsiX:
        self.psiX = C@A
        for k in range(self.p-1):
            self.psiX = np.block([[self.psiX], [self.psiX[-self.model.nY:]@A]])

        ## PsiU:
        self.psiU = C@B + D
        for k in range(1, self.p):
            self.psiU = np.block([[self.psiU], [C@np.linalg.matrix_power(A,k)@B + D]])

        ## ThetaX
        self.thetaX = np.zeros((self.p*self.model.nY, self.m*self.model.nU))
        for i in range(self.p):
            for j in range(self.m):
                if j == i + 1:
                    self.thetaX[i*self.model.nY:(i+1)*self.model.nY, j*self.model.nU:(j+1)*self.model.nU] = self.Dtil
                elif j <= i:
                    self.thetaX[i*self.model.nY:(i+1)*self.model.nY, j*self.model.nU:(j+1)*self.model.nU] = self.Ctil @ np.linalg.matrix_power(self.Atil, i-j) @ self.Btil
                else:
                    continue

        ## ThetaU
        self.thetaU = np.zeros((self.m*self.model.nU, self.m*self.model.nU))
        for i in range(self.m):
            for j in range(self.m):
                if j <= i:
                    self.thetaU[i*self.model.nU:(i+1)*self.model.nU, j*self.model.nU:(j+1)*self.model.nU] = np.eye(self.model.nU)
                else:
                    continue

        ## Psi
        self.psi = np.block([self.psiX,self.psiU])

        ## Qy
        self.Qtil = np.kron(np.eye(self.p), self.Q)
        
        ## R
        self.Rtil = np.kron(np.eye(self.m), self.R)

        ## L
        self.L = np.kron(np.ones((self.p, 1)), np.array([[0],[0],[0],[1]]))
        self.gamma = np.array([self.gamma]).reshape(-1,1)

        ## Esp
        self.Esp = np.array([-1]).reshape(-1,1)

        ## F
        ### Como a matriz leva x_k dentro dela, ela esta ja dentro do proprio MPC

        ## Hessiana
        self.H = np.block([[self.thetaX.T @ self.Qtil @ self.thetaX + self.Rtil + self.thetaX.T @ self.L @ self.gamma @ self.L.T @ self.thetaX, - self.thetaX.T @ self.Qtil],
                            [- self.Qtil @ self.thetaX, self.Qtil ]])
        self.H = (self.H + self.H.T)/2

        ## Ibar
        Ibar = np.kron(np.ones((self.m,1)), np.eye(self.model.nU))

        ## G
        self.G = np.block([[np.zeros((self.p * self.model.nY, self.m * self.model.nU)), np.eye(self.p * self.model.nY)],
                            [np.zeros((self.p * self.model.nY, self.m * self.model.nU)), - np.eye(self.p * self.model.nY)],
                            [self.thetaU,  np.zeros((self.m*self.model.nU, self.p * self.model.nY))],
                            [-self.thetaU,  np.zeros((self.m*self.model.nU, self.p * self.model.nY))],
                            [np.eye(self.m*self.model.nU),  np.zeros((self.m*self.model.nU, self.p * self.model.nY))],
                            [-np.eye(self.m*self.model.nU),  np.zeros((self.m*self.model.nU, self.p * self.model.nY))]])
        
        ## S
        self.S = np.block([[np.zeros((self.p * self.model.nY, self.model.nX)), np.zeros((self.p * self.model.nY, self.model.nU))],
                            [np.zeros((self.p * self.model.nY, self.model.nX)), np.zeros((self.p * self.model.nY, self.model.nU))],
                            [np.zeros((self.m*self.model.nU, self.model.nX)), - Ibar],
                            [np.zeros((self.m*self.model.nU, self.model.nX)), Ibar],
                            [np.zeros((self.m*self.model.nU, self.model.nX)), np.zeros((self.m*self.model.nU, self.model.nU))],
                            [np.zeros((self.m*self.model.nU, self.model.nX)), np.zeros((self.m*self.model.nU, self.model.nU))]])
        
        print('Controlador MPC construído.')
        
    def run(self, plant, iter, yspMaxList, yspMinList):
        self.iter = iter
        self.yspMaxList = yspMaxList
        self.yspMinList = yspMinList
        # Inicializando Variáveis para o EMPC

        ## Pontos iniciais
        xk = self.model.normalize(self.model.x_ss.copy(), 'x')
        uk = self.model.normalize(self.model.u_ss.copy(), 'u')
        x_plant = self.model.x_ss.copy()
        z_plant = self.model.z_ss.copy()
        u_plant = self.model.u_ss.copy()
        x_k = np.block([[xk],[uk]]) # Estado estentido [[x(k)], [u(k-1)]]

        # Filtro de Kalman
        KF = self.kalmanFilter()

        ## Upper & Lower Bounds
        uMax = np.tile(np.array([[0.43]]).reshape(-1,1), (self.m,1))
        uMin = np.tile(-np.array([[0.2]]).reshape(-1,1), (self.m,1))
        dUMax = np.tile(np.ones((self.model.nU,1))/20, (self.m,1))

        w = np.block([[self.yspMaxList[:,0].reshape(-1,1)],
                      [-self.yspMinList[:,0].reshape(-1,1)],
                      [uMax],
                      [-uMin],
                      [dUMax],
                      [dUMax]])

        ## Vetores para visualização
        deltaU_value = np.zeros((self.m*self.model.nU, iter))
        self.ysp_value = np.zeros((self.p*self.model.nY, iter))
        deltaU_mpc = np.zeros((self.model.nU, iter))
        self.y_value = np.zeros((self.model.nY, iter))
        self.y_mpc_value = np.zeros((self.model.nY, iter))
        self.u_value = np.zeros((self.model.nU, iter))
        
        # Loop do EMPC
        print(f'Iniciando simulação EMPC por {iter} iterações...')
        for k in range(iter):
            ## Mudança nas restrições do setpoint
            w[:self.p*self.model.nY] = self.yspMaxList[:,k].reshape(-1,1)
            w[self.p*self.model.nY:2*self.p*self.model.nY] = -self.yspMinList[:,k].reshape(-1,1)

            ## Criação do problema de otimização
            Z = cp.Variable((self.m*self.model.nU + self.p*self.model.nY,1))
            F = np.block([[x_k.T @ self.psi.T @ self.Qtil @ self.thetaX + x_k.T @ self.psi.T @ self.L @ self.gamma @ self.L.T @ self.thetaX - self.Esp @ self.gamma @ self.L.T @ self.thetaX, - x_k.T @ self.psi.T @ self.Qtil ]])
            cost = cp.quad_form(Z,self.H) + 2 * F @ Z
            constraints = [self.G @ Z <= self.S @ x_k + w]
            prob = cp.Problem(cp.Minimize(cost), constraints)
            # Desabilitando verbose para maior velocidade, habilite se necessário
            prob.solve(solver = "OSQP", verbose = False) 

            ## Coleta de resultados
            deltaU_value[:,k] = Z.value[:self.m*self.model.nU].flatten()
            deltaU_mpc[:,k] = deltaU_value[:self.model.nU, k]
            self.ysp_value[:,k] = Z.value[self.m*self.model.nU:].flatten()

            ysp_opt = Z.value[self.m*self.model.nU:].flatten()
            ysp_min_k = self.yspMinList[:self.model.nY, k]
            ysp_max_k = self.yspMaxList[:self.model.nY, k]
            
            print(f"\nIteração {k}:")
            print(f"ysp_min (normalizado): {ysp_min_k}")
            print(f"ysp_max (normalizado): {ysp_max_k}")
            print(f"ysp_opt (normalizado): {ysp_opt[:self.model.nY]}")
            print(f"Violação inferior? {any(ysp_opt[:self.model.nY] < ysp_min_k)}")
            print(f"Violação superior? {any(ysp_opt[:self.model.nY] > ysp_max_k)}")

            ## Passo na planta
            u_plant = u_plant + self.model.u_ss * deltaU_mpc[:,k].reshape(-1,1)
            # resPlant = plant.run(x_plant.flatten(), z_plant.flatten(), u_plant.flatten())
            # for i in range(plant.sistema.n_points):
            #     x_plant[i*3:(i+1)*3] = np.array([resPlant['T_sol'][:,i], resPlant['V_sol'][:,i], resPlant['w_sol'][:,i]])
            # z_plant = resPlant['z_sol'].T
            
            # Matrizes de visualização
            # self.y_value[:,k] = np.array([resPlant["T_sol"][:,0], resPlant["z10"], resPlant["z11"], resPlant["potencia"]]).flatten()
            self.u_value[:,k] = u_plant.flatten() 

            ## Passo a frente no modelo
            x_mpc = self.Atil @ x_k + self.Btil @ deltaU_mpc[:,k].reshape(-1,1)
            y_mpc = self.Ctil @ x_k + self.Dtil @ deltaU_mpc[:,k].reshape(-1,1)
            self.y_mpc_value[:,k] = y_mpc.flatten()

            ## Estimação de estados com filtro de Kalman
            x_k = x_mpc #+ KF @ (self.model.normalize(self.y_value[:,k], 'y') - y_mpc)
        
        print('Simulação concluída.')
        self.save_results()
        print(f'Resultados salvos em: {self.results_path}')

    def kalmanFilter(self, iter = 10000):
        covMedido = .9
        covSistema = .1
        sM = self.Atil.shape
        PP = np.eye(sM[1])
        VV = np.eye(self.model.nY)*covMedido
        WW = np.eye(sM[1])*covSistema
        for i in range(iter):
            # 1. Cálculo da Inovação Covariância (S)
            S = VV + self.Ctil @ PP @ self.Ctil.T
            
            # 2. Cálculo do Ganho de Kalman (K_F)
            # K_F = A*P*C^T * inv(S)
            KF_k = self.Atil @ PP @ self.Ctil.T @ np.linalg.inv(S)
            
            # 3. Atualização da Covariância P (P_next = A * P * A^T - K_F * S * K_F^T + W)
            # Usando P_next = A * (P - K_F * C * P) * A^T + W é mais estável
            
            # Pk_corrected = Pk - Pk * C^T * S^-1 * C * Pk
            P_k_corrected = PP - PP @ self.Ctil.T @ np.linalg.inv(S) @ self.Ctil @ PP
            
            # P_next = A @ P_corrected @ A^T + W
            PP = self.Atil @ P_k_corrected @ self.Atil.T + WW
        
        # O ganho de Kalman retornado deve ser o ganho do último passo (K_F_final)
        S_final = VV + self.Ctil @ PP @ self.Ctil.T
        KF_final = self.Atil @ PP @ self.Ctil.T @ np.linalg.inv(S_final)
        
        return KF_final

    def save_results(self):
        """Salva as variáveis de plotagem em um arquivo pickle."""
        data_to_save = {
            'iter': self.iter,
            'y_value': self.y_value,
            'u_value': self.u_value,
            'yspMaxList': self.yspMaxList,
            'yspMinList': self.yspMinList,
            'ysp_value': self.ysp_value,
            'y_mpc_value': self.y_mpc_value,
            'dt': self.model.dt,
        }
        output_dir = os.path.dirname(self.results_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(self.results_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_results(self, yspMaxList_new, yspMinList_new):
        """Carrega as variáveis de plotagem de um arquivo pickle."""
        try:
            with open(self.results_path, 'rb') as f:
                data = pickle.load(f)
            
            # Garante que os setpoints carregados correspondem aos setpoints que o usuário tentou rodar
            # Isto é crucial para que o plot dos limites faça sentido.
            if data['iter'] == yspMaxList_new.shape[1] and \
               np.array_equal(data['yspMaxList'], yspMaxList_new) and \
               np.array_equal(data['yspMinList'], yspMinList_new):
                
                self.iter = data['iter']
                self.y_value = data['y_value']
                self.u_value = data['u_value']
                self.yspMaxList = data['yspMaxList']
                self.yspMinList = data['yspMinList']
                self.ysp_value = data['ysp_value']
                self.y_mpc_value = data['y_mpc_value']
                print(f'Resultados carregados com sucesso de: {self.results_path}')
                return True
            else:
                print('Arquivo de resultados encontrado, mas não corresponde aos setpoints atuais. Rodando simulação.')
                return False
        except FileNotFoundError:
            print(f'Arquivo de resultados não encontrado: {self.results_path}. Rodando simulação.')
            return False
        except Exception as e:
            print(f'Erro ao carregar resultados ({e}). Rodando simulação.')
            return False

    def plot(self):
        # Plot das figuras de controle
        output_dir = "ENGG21 - Controle Avançado\EMPC\graficos"
        os.makedirs(output_dir, exist_ok=True)
        # O tempo agora depende do self.model.dt, que é implicitamente conhecido. 
        # O tempo está definido em horas (/3600)
        t = np.arange(self.iter) * self.model.dt / 3600

        ySPMaxDen = np.zeros((self.model.nY, self.iter))
        ySPMinDen = np.zeros((self.model.nY, self.iter))
        ySPDen = np.zeros((self.model.nY, self.iter))
        ympcDen = np.zeros((self.model.nY, self.iter))
        for i in range(self.iter):
            ySPMaxDen[:,i] = self.model.denormalize(self.yspMaxList[:self.model.nY, i], 'y').flatten()
            ySPMinDen[:,i] = self.model.denormalize(self.yspMinList[:self.model.nY, i], 'y').flatten()
            ySPDen[:,i] = self.model.denormalize(self.ysp_value[:self.model.nY, i], 'y').flatten()
            ympcDen[:,i] = self.model.denormalize(self.y_mpc_value[:, i], 'y').flatten()

        T2 = self.y_value[0]
        m_dot = self.y_value[1]
        P = self.y_value[2]
        W = self.y_value[3]

        rot = self.u_value[0]

        # --- Plots com Legenda Ajustada ---

        plt.figure(figsize=(20, 12))
        plt.plot(t, T2, label='Temperatura na saída do compressor / K', linewidth=5)
        #plt.plot(t, ympcDen[0], color = 'g')
        plt.ylabel('Temperatura na saída do compressor / K')
        plt.xlabel('Tempo / h')
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fancybox=True, shadow=True, fontsize=35)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(output_dir, 'EMPC_y0_temperaturaCompressor.png'))

        plt.figure(figsize=(20, 12))
        plt.plot(t, m_dot, label=r'Vazão Mássica / kg/s', linewidth=5)
        #plt.plot(t, ympcDen[1], color = 'g')
        plt.ylabel(r'Vazão Mássica / kg/s')
        plt.xlabel('Tempo / h')
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fancybox=True, shadow=True, fontsize=35)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(output_dir, 'EMPC_y1_vazaoMassica.png'))

        plt.figure(figsize=(20, 12))
        #plt.plot(t, P, label='Pressão no final do duto / kPa', linewidth=5)
        plt.plot(t, ySPMaxDen[2], label='Limites de SetPoint', linestyle='--', color='k', linewidth=5)
        plt.plot(t, ySPDen[2], label='SetPoint', linestyle='--', color='r', linewidth=5)
        plt.plot(t, ympcDen[2], color = 'g')
        plt.plot(t, ySPMinDen[2], linestyle='--', color='k', linewidth=5)
        plt.ylabel('Pressão no final do duto / kPa')
        plt.xlabel('Tempo / h')
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fancybox=True, shadow=True, fontsize=35)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(output_dir, 'EMPC_y2_pressaoDuto.png'))

        plt.figure(figsize=(20, 12))
        plt.plot(t, W, label='Potência do Compressor / kWh', linewidth=5)
        #plt.plot(t, ympcDen[3], color = 'g')
        plt.ylabel('Potência do Compressor / kWh')
        plt.xlabel('Tempo / h')
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=1, fancybox=True, shadow=True, fontsize=35)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(output_dir, 'EMPC_y3_potencia.png'))

        plt.figure(figsize=(20, 12))
        plt.plot(t, rot, label='Vel. Rotação do Compressor / hz', linewidth=5)
        plt.ylabel('Vel. Rotação do Compressor / hz')
        plt.xlabel('Tempo / h')
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=1, fancybox=True, shadow=True, fontsize=35)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(os.path.join(output_dir, 'EMPC_u0_rotação.png'))


if __name__ == "__main__":
    model = model2ss.linDiscretizeComp()

    # Configurações do Controlador:
    p = 10 # Horizonte de Predição
    m = 3 # Horizonte de Controle
    Q = np.diag([0,0,50,0]) # Peso das Saídas
    R = np.diag([1]) # Peso das Entradas
    gamma = 25 # Peso da Parcela Econômica
    iter = 60*6 # Pontos para simulação do controlador

    # Mudança nas restrições do set point (DEVE SER DEFINIDA ANTES DE TENTAR CARREGAR)
    yspMaxList = np.ones((p*model.nY, iter))/6
    yspMinList = -np.ones((p*model.nY, iter))/10

    for i in range(iter):
        yspMinList[2::model.nY, i] = -0.05
        if i > 6*12.5 and i <= 6*40:
            yspMinList[2::model.nY, i] = 0.0
        if i > 6*25 and i <= 6*40:
            yspMinList[2::model.nY, i] = 0.05
    
    EMPC = EMPC(model, p, m, Q, R, gamma)

    # Tenta carregar resultados salvos. Se falhar, roda a simulação.
    #if not EMPC.load_results(yspMaxList, yspMinList):
    EMPC.run(model.plant, iter, yspMaxList, yspMinList)
        
    EMPC.plot()