import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import simulation as sim

class NMPC:
    def __init__(self, p, m, steps, nY, nX, nU, Q, R, dt, SP):
        self.p = p
        self.m = m
        self.steps = steps
        self.nU = nU
        self.nX = nX
        self.nY = nY
        self.Q = Q
        self.R = R

        self.sim_pred = sim.RiserModel(p, m, steps, dt)
        self.sim_mf = sim.RiserModel(1, 1, steps, dt)

        self.x_sp = self.sim_mf.setPoints(SP)

        # TODO: Adicionar restrições de entrada e estado
        self.u_min = np.array([0, 0])
        self.u_max = np.array([20, 20])
        self.dU_min = np.array([-0.05, -0.05])
        self.dU_max = np.array([0.05, 0.05])
        self.x_min = np.array([0, 0, 0, 0, 0, 0, 0, 0])

