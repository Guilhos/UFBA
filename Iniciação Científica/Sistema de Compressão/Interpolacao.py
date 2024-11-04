import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class DataInterpolator:
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
        self.X_sample = np.sort(self.data.columns.values)
        self.Y_sample = np.arange(len(self.data))
        self.Z_sample = self.data.values
        print("Dados carregados e amostrados aleatoriamente com sucesso.")
        print("Dimensão de X_sample:", self.X_sample.shape)
        print("Dimensão de Y_sample:", self.Y_sample.shape)
        print("Dimensão de Z_sample:", self.Z_sample.shape)

    def interpolate(self, num_points=100):
        """Executa a interpolação bilinear nos dados amostrados."""
        # Criar uma malha densa para interpolação
        x_dense = np.linspace(self.X_sample.min(), self.X_sample.max(), num_points)
        y_dense = np.linspace(self.Y_sample.min(), self.Y_sample.max(), num_points)
        X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

        # Executar a interpolação
        points = np.array([(x, y) for y in self.Y_sample for x in self.X_sample])  # Pontos da malha original
        values = self.Z_sample.flatten()  # Valores correspondentes aos pontos
        Z_dense = griddata(points, values, (X_dense, Y_dense), method='linear')  # Armazenar em Z_dense

        print("Dimensão de X_dense:", X_dense.shape)
        print("Dimensão de Y_dense:", Y_dense.shape)
        print("Dimensão de Z_dense:", Z_dense.shape)
        
        return X_dense, Y_dense, Z_dense

    def plot_results(self, X_dense, Y_dense, Z_dense):
        """Plota os resultados da interpolação e da amostra original."""
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

        # Dados Originais (Amostra)
        X_grid, Y_grid = np.meshgrid(self.X_sample, self.Y_sample)
        ax1.scatter(X_grid, Y_grid, self.Z_sample, cmap='viridis', edgecolor='k')
        ax1.set_title("Dados Originais (Amostra)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Superfície Interpolada
        ax2.plot_surface(X_dense, Y_dense, Z_dense, cmap='viridis')
        ax2.set_title("Superfície Interpolada (Bilinear)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    file_path = 'e:/Faculdade/UFBA/UFBA-1/Iniciação Científica/Sistema de Compressão/tabela_phi.csv'
    perc = 0.5
    interpolator = DataInterpolator(file_path)
    interpolator.load_data(perc)
    X_dense, Y_dense, Z_dense = interpolator.interpolate(num_points=100)
    interpolator.plot_results(X_dense, Y_dense, Z_dense)
