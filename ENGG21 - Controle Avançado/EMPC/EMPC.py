import numpy as np
import control as ctrl 
import matplotlib.pyplot as plt
import cvxpy as cp
import model2ss
import os

class EMPC:
    def __init__(self, model, p, m, Q, R):
        self.p = p
        self.m = m
        self.Q = Q
        self.R = R
        self.model = model
        self.buildController()

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
        self.thetaX = np.zeros((p*self.model.nY, m*self.model.nU))
        for i in range(self.p):
            for j in range(self.m):
                if j == i + 1:
                    self.thetaX[i*self.model.nY:(i+1)*self.model.nY, j*self.model.nU:(j+1)*self.model.nU] = self.Dtil
                elif j <= i:
                    self.thetaX[i*self.model.nY:(i+1)*self.model.nY, j*self.model.nU:(j+1)*self.model.nU] = self.Ctil @ np.linalg.matrix_power(self.Atil, i-j) @ self.Btil
                else:
                    continue

        ## ThetaU
        self.thetaU = np.zeros((m*self.model.nU, m*self.model.nU))
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

        ## Hessiana
        self.H = self.thetaX.T @ self.Qtil @ self.thetaX + self.Rtil
        self.H = (self.H + self.H.T)/2

        ## F
        self.F = np.block([[self.psi.T @ self.Qtil @ self.thetaX],[np.kron(np.ones((p,1)), np.eye(self.model.nY)).T @ - self.Qtil @ self.thetaX]])

        ## Ibar
        Ibar = np.kron(np.ones((m,1)), np.eye(self.model.nU))

        ## G
        self.G = np.block([[self.thetaX],
                     [-self.thetaX],
                     [self.thetaU],
                     [-self.thetaU],
                     [np.eye(self.m*self.model.nU)],
                     [-np.eye(self.m*self.model.nU)]])
        
        ## S
        self.S = np.block([[-self.psiX, -self.psiU],
                     [self.psiX, self.psiU],
                     [np.zeros((self.m*self.model.nU, self.model.nX)), - Ibar],
                     [np.zeros((self.m*self.model.nU, self.model.nX)), Ibar],
                     [np.zeros((self.m*self.model.nU, self.model.nX)), np.zeros((self.m*self.model.nU, self.model.nU))],
                     [np.zeros((self.m*self.model.nU, self.model.nX)), np.zeros((self.m*self.model.nU, self.model.nU))]])
        
    def run(self, plant, iter, y_spList):
        self.iter = iter
        self.y_spList = y_spList
        # Inicializando Variáveis para o EMPC

        ## Pontos iniciais
        y_sp = self.model.normalize(self.model.y_ss.copy(), 'y')
        xk = self.model.normalize(self.model.x_ss.copy(), 'x')
        uk = self.model.normalize(self.model.u_ss.copy(), 'u')
        x_plant = self.model.x_ss.copy()
        z_plant = self.model.z_ss.copy()
        u_plant = self.model.u_ss.copy()
        x_k = np.block([[xk],[uk]]) # Estado estentido [[x(k)], [u(k-1)]]
        z_k = np.block([[x_k],[y_sp]]) # Estado da parcela linear estendido [[x_(k)], [y_sp]]

        # Filtro de Kalman
        KF = self.kalmanFilter()

        ## Upper & Lower Bounds
        yMax = np.tile(np.ones((self.model.nY,1)), (self.p,1))
        yMin = np.tile(-np.ones((self.model.nY,1)), (self.p,1))
        uMax = np.tile(np.ones((self.model.nU,1)), (self.m,1))
        uMin = np.tile(-np.ones((self.model.nU,1)), (self.m,1))
        dUMax = np.tile(np.ones((self.model.nU,1))/5, (self.m,1))

        w = np.block([[yMax],
                      [-yMin],
                      [uMax],
                      [-uMin],
                      [dUMax],
                      [dUMax]])

        ## Vetores para visualização
        deltaU_value = np.zeros((self.m*self.model.nU, iter))
        deltaU_mpc = np.zeros((self.model.nU, iter))
        self.y_value = np.zeros((self.model.nY, iter))
        self.u_value = np.zeros((self.model.nU, iter))
        
        # Loop do EMPC
        for k in range(iter):
            ## Criação do problema de otimização
            deltaU = cp.Variable((self.m*self.model.nU,1))
            cost = cp.quad_form(deltaU,self.H) + 2 * z_k.T @ self.F @ deltaU
            constraints = [self.G @ deltaU <= self.S @ x_k + w]
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver = "OSQP", verbose = True)

            ## Coleta de resultados
            deltaU_value[:,k] = deltaU.value.flatten()
            deltaU_mpc[:,k] = deltaU_value[:self.model.nU, k]

            ## Perturbação no setpoint
            y_sp = y_spList[:,k].reshape(-1,1)

            ## Passo na planta
            u_plant = u_plant + self.model.u_ss * deltaU_mpc[:,k].reshape(-1,1)
            resPlant = plant.run(x_plant.flatten(), z_plant.flatten(), u_plant.flatten())
            for i in range(plant.sistema.n_points):
                x_plant[i*3:(i+1)*3] = np.array([resPlant['T_sol'][:,i], resPlant['V_sol'][:,i], resPlant['w_sol'][:,i]])
            z_plant = resPlant['z_sol'].T
            
            # Matrizes de visualização
            self.y_value[:,k] = np.array([resPlant["T_sol"][:,0], resPlant["z10"], resPlant["z11"]]).flatten()
            self.u_value[:,k] = u_plant.flatten() 

            ## Passo a frente no modelo
            x_mpc = self.Atil @ x_k + self.Btil @ deltaU_mpc[:,k].reshape(-1,1)
            y_mpc = self.Ctil @ x_k + self.Dtil @ deltaU_mpc[:,k].reshape(-1,1)

            if k == 5:
                print("test")

            ## Estimação de estados com filtro de Kalman
            x_k = x_mpc + KF @ (self.model.normalize(self.y_value[:,k], 'y') - y_mpc)
            z_k = np.block([[x_k], [y_sp]])

    def kalmanFilter(self, iter = 100):
        covMedido = .5
        covSistema = .5
        sM = self.Atil.shape
        PP = np.eye(sM[1])
        VV = np.eye(self.model.nY)*covMedido
        WW = np.eye(sM[1])*covSistema
        for i in range(iter):
            PP = self.Atil @ PP @ self.Atil.T - self.Atil @ PP @ self.Ctil.T @ np.linalg.inv(VV + self.Ctil @ PP @ self.Ctil.T) @ self.Ctil @ PP @ self.Atil.T + WW
        
        return self.Atil @ PP @ self.Ctil.T @ np.linalg.inv(VV + self.Ctil @ PP @ self.Ctil.T)

    def plot(self):
        # Plot das figuras de controle
        output_dir = "ENGG21 - Controle Avançado\EMPC\graficos"
        os.makedirs(output_dir, exist_ok=True)
        t = np.arange(self.iter) * self.model.dt

        y_spListDenorm = self.y_spList.copy()
        for i in range(iter):
            y_spListDenorm[:, i] = self.model.denormalize(self.y_spList[:, i], 'y').flatten()

        T2 = self.y_value[-3]
        m_dot = self.y_value[-2]
        P = self.y_value[-1]

        rot = self.u_value[0]
        P1 = self.u_value[1]
        T1 = self.u_value[2]
        Vv = self.u_value[3]

        plt.figure(figsize=(10, 6))
        plt.plot(t,T2, label='Temperatura na saída do compressor / K')
        plt.plot(t,y_spListDenorm[0], label = 'SetPoint')
        plt.ylabel('Temperatura na saída do compressor / K')
        plt.xlabel('Tempo / s')
        plt.savefig(os.path.join(output_dir, 'MPC_temperaturaCompressor.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(t,m_dot, label=r'Vazão Mássica / /frac{kg}{s}')
        plt.plot(t,y_spListDenorm[1], label = 'SetPoint')
        plt.ylabel(r'Vazão Mássica / /frac{kg}{s}')
        plt.xlabel('Tempo / s')
        plt.savefig(os.path.join(output_dir, 'MPC_vazaoMassica.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(t,P, label='Pressão no final do duto / MPa')
        plt.plot(t,y_spListDenorm[2], label = 'SetPoint')
        plt.ylabel('Pressão no final do duto / MPa')
        plt.xlabel('Tempo / s')
        plt.savefig(os.path.join(output_dir, 'MPC_pressaoDuto.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(t,rot, label='Pressão no final do duto / MPa')
        plt.ylabel('Rotação Compressor')
        plt.xlabel('Tempo / s')
        plt.savefig(os.path.join(output_dir, 'MPC_rotação.png'))


if __name__ == "__main__":
    model = model2ss.linDiscretizeComp()

    # Configurações do Controlador:
    p = 10 # Horizonte de Predição
    m = 3 # Horizonte de Controle
    Q = np.diag([1,1,10]) # Peso das Saídas
    R = np.eye(model.nU) # Peso das Entradas
    iter = 10 # Pontos para simulação do controlador

    y_spList = np.zeros((model.nY, iter))
    for i in range(iter):
        if i >= 2:
            y_spList[-2,i] = y_spList[-2,i] - 0.1
        if i >= 7:
            y_spList[-1,i] = y_spList[-1,i] + 0.1
    
    EMPC = EMPC(model, p, m, Q, R)
    EMPC.run(model.plant,iter,y_spList)
    EMPC.plot()
