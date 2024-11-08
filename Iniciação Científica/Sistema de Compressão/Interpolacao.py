import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class DataInterpolatorCasadi:
    def __init__(self, file_path, decimal=','):
        self.file_path = file_path
        self.decimal = decimal
        self.data = None
        self.X_sample = None
        self.Y_sample = None
        self.Z_sample = None

    def load_data(self, perc):
        """Carrega os dados do arquivo CSV, organiza as colunas e amostra 1% dos dados aleatoriamente."""
        self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        self.data.columns = self.data.columns.str.replace(',', '.').astype(float)

        # Transpor para garantir que cada coluna represente uma amostra
        self.X_sample = np.arange(len(self.data)) # Vai de 0 até 39, Shape: (40,)
        self.Y_sample = np.sort(self.data.columns.values) # Vai de -2.80 até 1.11, Shape: (181,)
        self.Z_sample = self.data.values
        print("Dados carregados e amostrados aleatoriamente com sucesso.")
        print("Dimensão de X_sample:", self.X_sample.shape)
        print("Dimensão de Y_sample:", self.Y_sample.shape)
        print("Dimensão de Z_sample:", self.Z_sample.shape)

    def interpolate(self, num_points=100):
        """Executa a interpolação bilinear nos dados amostrados com CasADi."""
        # Criar uma malha densa para interpolação
        x_dense = np.linspace(self.X_sample.min(), self.X_sample.max(), num_points)
        y_dense = np.linspace(self.Y_sample.min(), self.Y_sample.max(), num_points)
        z_flat = self.Z_sample.ravel(order='F')

        lut = ca.interpolant('name','bspline',[self.X_sample, self.Y_sample],z_flat)

        # Calcular a malha de Z usando os pontos interpolados
        Z_dense = np.zeros((num_points, num_points))
        for i, x in enumerate(x_dense):
            for j, y in enumerate(y_dense):
                Z_dense[i, j] = lut([x, y])

        print("Dimensão de Z_dense:", Z_dense.shape)
        return x_dense, y_dense, Z_dense, lut

    def plot_results(self, X_dense, Y_dense, Z_dense,x_test,y_test,z_interpolado):
        """Plota os resultados da interpolação e da amostra original."""
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

        # Dados Originais (Amostra)
        X_grid, Y_grid = np.meshgrid(self.Y_sample, self.X_sample) # EU REALMENTE NAO SEI COMO EU ENTRO COM 181,40 E O MESHGRID SAI COMO 40,181
        print(X_grid.shape,Y_grid.shape)
        ax1.scatter(X_grid, Y_grid, self.Z_sample,c=self.Z_sample.ravel(), cmap='viridis', edgecolor='k')
        ax1.set_title("Dados Originais (Amostra)")
        ax1.set_xlabel("N")
        ax1.set_ylabel("Vazão")
        ax1.set_zlabel("Phi")

        # Superfície Interpolada
        X_dense_grid, Y_dense_grid = np.meshgrid(Y_dense, X_dense)
        ax2.plot_surface(X_dense_grid, Y_dense_grid, Z_dense, cmap='viridis')
        ax2.set_title("Superfície Interpolada (CasADi)")
        ax2.set_xlabel("N")
        ax2.set_ylabel("Vazão")
        ax2.set_zlabel("Phi")

        ax2.scatter(y_test,x_test,z_interpolado, c = 'black')

        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    interpol = DataInterpolatorCasadi('E:/Faculdade/UFBA/UFBA/Iniciação Científica/Sistema de Compressão/tabela_phi.csv')
    interpol.load_data(0.5)

    # Cria a função de interpolação
    X_dense, Y_dense, Z_dense, interpolant_func = interpol.interpolate(num_points=100)

    # Testa a função de interpolação
    x_test = 25  # Exemplo de ponto PODE IR DE 0 A 39, acho que X é provavelmente a massa
    y_test = 1  # Exemplo de ponto PODE IR DE -3 ATE 1.11, y deve ser o N
    # EU REALMENTE NAO SEI COMO AS COORDENADAS ESTAO FUNCIONANDO NA HORA DE PLOTAR, MAS FUNCIONAM, NAO ME PERGUNTE
    z_interpolado = interpolant_func([x_test, y_test])
    print(f"Valor interpolado em (x={x_test}, y={y_test}):", z_interpolado)

    interpol.plot_results(X_dense, Y_dense, Z_dense,x_test,y_test,z_interpolado)
