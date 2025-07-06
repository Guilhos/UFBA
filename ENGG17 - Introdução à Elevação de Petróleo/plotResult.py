import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_results(iter, Ymk, Ypk, Upk, dU, Ysp, Tempos, dt):
    # Verifica se o diretório 'plots' existe, caso contrário, cria
    imagesPath = os.path.join(os.getcwd(), 'ENGG17 - Introdução à Elevação de Petróleo/plots')
    os.makedirs(imagesPath, exist_ok=True)

    # Plot das entradas e pressões de fundo de poço
    plt.figure(figsize=(16, 14))

    Ymk = np.array(Ymk)
    Ypk = np.array(Ypk)
    Upk = np.array(Upk)
    dU = np.array(dU)
    Ysp = np.array(Ysp)

    t = np.arange(iter)*dt

    plt.subplot(2, 1, 1)
    plt.plot(t, Upk[:, 0], label="wgl1 (Poço 1)")
    plt.plot(t, Upk[:, 1], label="wgl2 (Poço 2)")
    plt.ylabel("Injeção de gás [kg/s]")
    plt.title("Controles aplicados")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, Ypk[:, 0], label="Pbh Poço 1 (Pressão de Fundo)")
    plt.plot(t, Ypk[:, 1], label="Pbh Poço 2 (Pressão de Fundo)")
    plt.plot(t, Ymk[:, 0], linestyle='-.', color='blue')
    plt.plot(t, Ymk[:, 1], linestyle='-.', color='orange')
    plt.plot(t, Ysp[:, 0], linestyle='--', color='blue', label="Pressão de Fundo Esperada Poço 1")
    plt.plot(t, Ysp[:, 1], linestyle='--', color='orange', label="Pressão de Fundo Esperada")
    plt.ylabel("Pressão [Pa]")
    plt.xlabel("Tempo [s]")
    plt.title("Pressão de Fundo dos Poços")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "saida.png"))

with open('ENGG17 - Introdução à Elevação de Petróleo/results_NMPC.pkl', 'rb') as f:
    iter, Ymk, Ypk, Upk, dU, Ysp, Tempos, dt = pickle.load(f)

plot_results(iter, Ymk, Ypk, Upk, dU, Ysp, Tempos, dt)