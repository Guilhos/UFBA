import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import simulation as sim
import time

class NMPC:
    def __init__(self, p, m, steps, nY, nX, nU, Q, R, dt, SP, iter):
        self.p = p
        self.m = m
        self.steps = steps
        self.nU = nU
        self.nX = nX
        self.nY = nY
        self.Q = Q
        self.R = R
        self.iter = iter

        self.dU = np.zeros((nU * m, 1)) 

        self.sim_pred = sim.RiserModel(p, m, steps, nY, nX, nU, dt)
        self.sim_mf = sim.RiserModel(1, 1, steps, nY, nX, nU, dt)
        
        self.SPList = self.sim_mf.setPoints(SP)
        self.y_sp, _, _ = self.SPList[0]

        # TODO: Adicionar restrições de entrada e estado
        self.u_min = np.array([[0], [0]])
        self.u_max = np.array([[5], [5]])
        self.dU_min = np.array([[-0.1], [-0.1]])
        self.dU_max = np.array([[0.1], [0.1]])
        self.y_min = np.array([[0] for _ in range(nY)])
        self.y_max = np.array([[np.inf] for _ in range(nY)])
    
    def iTil(self, n, x):
        n = np.tile(n,(x,1))
        return n
    
    def diagMatrix(self, x,n):
        x = np.float64(x)
        n = int(n)
        X_matrix = np.full((n,n),0, dtype=np.float64)
        np.fill_diagonal(X_matrix,x)
        return X_matrix
    
    def matriz_triangular_identidade(self, m, n, N):
        matriz = np.zeros((m * N, n * N))
        
        for i in range(m):
            for j in range(n):
                if j <= i:
                    matriz[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.eye(N)
        
        return ca.DM(matriz)  # Convertendo para CasADi DM
    
    def ajusteMatrizes(self):
        self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1)) # Expansão do y_setpoint para P. SHAPE -> (nY*P, 1) # Expansão do y_min para P. SHAPE -> (nY*P, 1)
        self.y_max = ca.DM(self.iTil(self.y_max,self.p)) # Expansão do y_max para P. SHAPE -> (nY*P, 1)
        self.y_min = ca.DM(self.iTil(self.y_min,self.p)) # Expansão do y_min para P. SHAPE -> (nY*P, 1)

        self.u_min = ca.DM(self.iTil(self.u_min,self.m)) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
        self.u_max = ca.DM(self.iTil(self.u_max,self.m)) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

        self.dU_min = self.iTil(self.dU_min, self.m) # Expansão do dU_min para M. SHAPE -> (nU*M, 1)
        #self.dU_min = ca.DM(np.concatenate((self.dU_min,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)
        self.dU_max = self.iTil(self.dU_max, self.m) # Expansão do dU_max para M. SHAPE -> (nU*M, 1)
        #self.dU_max = ca.DM(np.concatenate((self.dU_max,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)

        # Supondo que self.Q tem shape (nY, 1)
        q_tile = np.array(self.Q * self.p)  # Repete p vezes → shape (nY*p,)
        self.Q = ca.DM(np.diag(q_tile))  # Cria matriz diagonal (nY*p, nY*p)

        r_tile = np.array(self.R * self.m)  # Repete m vezes → shape (nU*m,)
        self.R = ca.DM(np.diag(r_tile))  # Cria matriz diagonal (nU*m, nU*m)

    def nlp_func(self):
        # Criando o problema de otimização
        opti = ca.Opti()

        # Definição das variáveis de decisão
        dUs = opti.variable(self.nU * self.m, 1)
        Fs  = opti.variable(1, 1)  # Variável escalar para Fs
        yModelk = opti.parameter(self.nY * self.steps, 1)
        xModelk = opti.parameter(self.nX * self.steps, 1)  # xModelk como parâmetro
        uModelk = opti.parameter(self.nU * self.steps, 1)
        yPlantak = opti.parameter(self.nY, 1)
        ysp = opti.parameter(self.nY * self.p, 1)# yPlantak como parâmetro 
    
        x = ca.vertcat(dUs, Fs)
        
        # Definição do problema de otimização
        opti.minimize(Fs)

        # Erro entre a planta e o modelo
        dYk = yPlantak - yModelk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)

        # Predição do modelo
        yModel_pred, _, _ = self.sim_pred.caPredFun(xModelk, uModelk, dUs)

        # Matriz triangular para os controles
        matriz_inferior = self.matriz_triangular_identidade(self.m, self.m, self.nU)

        # Restrições
        # x_min e x_max
        opti.subject_to(opti.bounded(self.dU_min, dUs, self.dU_max))
        opti.subject_to(opti.bounded(0, Fs, 10e23))

        # lbg e ubg
        opti.subject_to(opti.bounded(self.y_min, yModel_pred + dYk, self.y_max))
        opti.subject_to(opti.bounded(self.u_min, ca.repmat(uModelk[-self.nU:], self.m, 1) + matriz_inferior @ dUs, self.u_max))
        opti.subject_to(Fs - ((yModel_pred - ysp + dYk).T @ self.Q @ (yModel_pred - ysp + dYk) + dUs.T @ self.R @ dUs) == 0)  # Restrições de igualdade

        opti.solver('ipopt', {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-6,                      # Tolerância do solver (pode ajustar entre 1e-4 e 1e-8)
            "ipopt.constr_viol_tol": 1e-8,          
            "ipopt.max_iter": 750,                   # Reduz número de iterações (ajustável)
            "ipopt.mu_strategy": "adaptive",         # Estratégia de barreira mais eficiente
            "ipopt.linear_solver": "mumps",          # Solver linear mais rápido para problemas médios/grandes
            "ipopt.sb": "yes"
        })
        print(opti)

        # Criando a função otimizada
        return opti.to_function(
            "opti_nlp",
            [yModelk, xModelk, uModelk, yPlantak, ysp, dUs, Fs],
            [x]
        )

    def otimizar(self, ymk, xmk, umk, ypk):
        dYk = ypk - ymk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)
        dU_init = self.dU
        yModel_init, _, _ = self.sim_pred.caPredFun(xmk, umk, dU_init)
        yModel_init = np.array(yModel_init.full())
        Fs_init = (yModel_init - self.y_sp + dYk).T @ self.Q @ (yModel_init - self.y_sp + dYk) + dU_init.T @ self.R @ dU_init

        x_opt = self.opti_nlp(ymk, xmk, umk, ypk, self.y_sp, dU_init, Fs_init)

        dU_opt = x_opt[:self.nU * self.m]
        dU_opt = np.array(dU_opt.full())
        return dU_opt
    
    def run(self):
        self.ajusteMatrizes()
        ymk, xmk, umk = self.sim_pred.pIniciais()
        ypk = ymk[-self.nY:]
        self.y_sp = ypk
        self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
        ymk_next = ypk
        xpk = xmk[-self.nX:]
        xmk_next = xpk
        upk = umk[-self.nU:]
        umk_next = upk

        self.opti_nlp = self.nlp_func()

        Ypk_forPrint = []
        Upk_forPrint = []
        dU_forPrint = []
        Ymk_forPrint = []
        Ysp_forPrint = []
        Tempos_forPrint = []

        change = self.iter // len(self.SPList)
        for i in range(self.iter): 
            t1 = time.time()
            print(15*'='+ f'Iteração {i+1}' + 15*'=')
            dU_opt = self.otimizar(ymk, xmk, umk, ypk)
            
            self.dUk = dU_opt[:self.nU]
            self.dU = dU_opt
            
            umk = umk.reshape(self.steps*self.nU, 1)
            umk = np.append(umk, umk[-self.nU:] + self.dUk)
            umk = umk[self.nU:]

            ymk_next, xmk_next, umk_next = self.sim_mf.caPredFun(xmk[-self.nX:], umk[-self.nU:], [0,0])
            ymk_next = np.array(ymk_next.full())
            xmk_next = np.array(xmk_next.full())
            umk_next = np.array(umk_next.full())
            
            t2 =  time.time()
            Tempos_forPrint.append(t2-t1)
            print(f'Tempo decorrido: {t2-t1}')
            
            ypk, xpk, upk = self.sim_mf.pPlanta(xpk, self.dUk)

            print('dYk: ',ymk_next - ypk)
            
            upk = upk.flatten()
            xpk = xpk.flatten()
            ypk = ypk.flatten()
            
            ymk = np.append(ymk, ymk_next)
            ymk = ymk[self.nY:]
            xmk = np.append(xmk, xmk_next)
            xmk = xmk[self.nX:]
            umk = np.append(umk, umk_next)
            umk = umk[self.nU:]

            Ymk_forPrint.append(ymk_next)
            Ypk_forPrint.append(ypk)
            Upk_forPrint.append(upk)
            dU_forPrint.append(self.dUk)
            print('dUk: ',dU_opt[:self.m*self.nU])
            Ysp_forPrint.append(self.y_sp)
            
            index = i // change
            if i >= 20:
                for j in range(self.p):
                    self.y_sp[2*j] = 9.5e6
                    self.y_sp[2*j+1] = 10e6
                
        #self.plot_results(iter, Ymk, Ypk, Upk, YspM, YspP, Tempos)
        
        return self.iter, Ymk_forPrint, Ypk_forPrint, Upk_forPrint, dU_forPrint, Ysp_forPrint, Tempos_forPrint

