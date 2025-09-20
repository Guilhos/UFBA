import numpy as np
import casadi as ca

class EKF:
    def __init__(self, f_model, h_model, nX, nY, nU, Q, R):
        """
        f_model: função CasADi que gera o próximo estado xk+1
        h_model: função CasADi que gera a saída yk a partir de xk
        nX: número de estados
        nY: número de saídas
        Q: covariância do processo (ruído do modelo)
        R: covariância da medição (ruído dos sensores)
        """
        self.nX = nX
        self.nY = nY
        self.Q = Q
        self.R = R

        # Criar variáveis simbólicas
        x = ca.MX.sym('x', nX)
        u = ca.MX.sym('u', nU)  # pega a dimensão da entrada u

        # Define funções para predição e saída
        self.f = f_model
        self.h = h_model

        # Jacobianos
        A = ca.jacobian(self.f(x, u), x)
        C = ca.jacobian(self.h(x, u), x)

        self.A_func = ca.Function('A', [x, u], [A])
        self.C_func = ca.Function('C', [x, u], [C])

        # Estado inicial
        self.x_hat = np.zeros((nX,))
        self.P = np.eye(nX)

    def step(self, u, y_meas):
        x_hat_ca = ca.DM(self.x_hat)
        u_ca = ca.DM(u)

        # Predição
        x_pred = np.array(self.f(x_hat_ca, u_ca).full())
        A = np.array(self.A_func(x_hat_ca, u_ca).full())
        P_pred = A @ self.P @ A.T + self.Q

        # Atualização (usando x_pred para calcular C)
        C = np.array(self.C_func(ca.DM(x_pred), u_ca).full())
        y_pred = np.array(self.h(ca.DM(x_pred), u_ca).full())

        S = C @ P_pred @ C.T + self.R
        S_inv = np.linalg.inv(S + 1e-6 * np.eye(self.nY))  # regularização
        K = P_pred @ C.T @ S_inv

        innovation = y_meas - y_pred
        self.x_hat = x_pred + K @ innovation
        self.P = (np.eye(self.nX) - K @ C) @ P_pred

        return self.x_hat.copy(), self.P.copy()

    def reset(self, x0=None):
        """Zera o filtro ou redefine o estado inicial"""
        self.x_hat = np.zeros((self.nX,)) if x0 is None else x0.copy()
        self.P = np.eye(self.nX)