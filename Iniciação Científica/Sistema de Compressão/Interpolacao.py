import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class DataInterpolatorCasadi:
    def __init__(self, file_path, decimal=','):
        self.file_path = file_path
        self.decimal = decimal
        self.data = None
        self.N_rot = None
        self.Mass = None
        self.Phi = None

    def load_data(self, perc):
        self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        
        self.N_rot = np.arange(2e4,6e4,1e3) # Vai de 20000hz até 60000hz, Shape: (40,)
        self.Mass = np.arange(3,21.1,0.1) # Vai de 3 até 21, Shape: (181,)
        self.Phi = self.data.values # Valores da tabela, Shape: (40,181)
        print("Dados carregados com sucesso.")
        print("Dimensão de N_rot:", self.N_rot.shape)
        print("Dimensão de Mass:", self.Mass.shape)
        print("Dimensão de Phi:", self.Phi.shape)

    def interpolate(self, num_points=100):
        # Criar uma malha densa para interpolação
        N_dense = np.linspace(self.N_rot.min(), self.N_rot.max(), num_points)
        Mass_dense = np.linspace(self.Mass.min(), self.Mass.max(), num_points)
        phi_flat = self.Phi.ravel(order='F')

        lut = ca.interpolant('name','bspline',[self.N_rot, self.Mass],phi_flat)

        # Calcular a malha de Z usando os pontos interpolados
        Phi_dense = np.zeros((num_points, num_points))
        for i, x in enumerate(N_dense):
            for j, y in enumerate(Mass_dense):
                Phi_dense[i, j] = lut([x, y])

        print("Dimensão de Phi_dense:", Phi_dense.shape)
        return N_dense, Mass_dense, Phi_dense, lut

    def plot_results(self, N_dense, Mass_dense, Phi_dense,x_test,y_test,z_interpolado):
        """Plota os resultados da interpolação e da amostra original."""
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

        # Dados Originais (Amostra)
        X_grid, Y_grid = np.meshgrid(self.Mass, self.N_rot) # EU REALMENTE NAO SEI COMO EU ENTRO COM 181,40 E O MESHGRID SAI COMO 40,181
        print(X_grid.shape,Y_grid.shape)
        ax1.scatter(X_grid, Y_grid, self.Phi,c=self.Phi.ravel(), cmap='viridis', edgecolor='k')
        ax1.set_title("Dados Originais (Amostra)")
        ax1.set_xlabel("Vazão")
        ax1.set_ylabel("N")
        ax1.set_zlabel("Phi")

        # Superfície Interpolada
        X_dense_grid, Y_dense_grid = np.meshgrid(Mass_dense, N_dense)
        ax2.plot_surface(X_dense_grid, Y_dense_grid, Phi_dense, cmap='viridis')
        ax2.set_title("Superfície Interpolada (CasADi)")
        ax2.set_xlabel("Vazão")
        ax2.set_ylabel("N")
        ax2.set_zlabel("Phi")

        ax2.scatter(y_test,x_test,z_interpolado, c = 'black')

        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    interpol = DataInterpolatorCasadi('/home/guilhermefreire/UFBA/Iniciação Científica/Sistema de Compressão/tabela_phi.csv')
    interpol.load_data(0.5)

    # Cria a função de interpolação
    N_dense, Mass_dense, Phi_dense, interpolant_func = interpol.interpolate(num_points=100)

    # Testa a função de interpolação
    N_test = 30000  # Exemplo de ponto PODE IR DE 20000 A 60000, acho que X é provavelmente a massa
    Mass_test = 15  # Exemplo de ponto PODE IR DE 3 ATE 21, y deve ser o N
    # EU REALMENTE NAO SEI COMO AS COORDENADAS ESTAO FUNCIONANDO NA HORA DE PLOTAR, MAS FUNCIONAM, NAO ME PERGUNTE
    z_interpolado = interpolant_func([N_test, Mass_test])
    print(f"Valor interpolado em (x={N_test}, y={Mass_test}):", z_interpolado)

    interpol.plot_results(N_dense, Mass_dense, Phi_dense,N_test,Mass_test,z_interpolado)
