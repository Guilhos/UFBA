import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_results(iter, Ymk, Ypk, Upk, dU, Ysp, Tempos):
    # Verifica se o diretório 'plots' existe, caso contrário, cria
    imagesPath = os.path.join(os.getcwd(), 'ENGG17 - Introdução à Elevação de Petróleo/plots')
    os.makedirs(imagesPath, exist_ok=True)

    plt.figure(figsize=(16, 14))

    Ymk = np.array(Ymk)
    Ypk = np.array(Ypk)
    Upk = np.array(Upk)
    dU = np.array(dU)
    Ysp = np.array(Ysp)

    t = np.arange(iter)

    plt.subplot(4, 1, 1)
    plt.plot(t, Upk[:, 0], label="wgl1 (Poço 1)")
    plt.plot(t, Upk[:, 1], label="wgl2 (Poço 2)")
    plt.ylabel("Injeção de gás [kg/s]")
    plt.title("Controles aplicados")
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, Ymk[:, 4], label="Óleo produzido Poço 1")
    plt.plot(t, Ymk[:, 14], label="Óleo produzido Poço 2")
    plt.plot(t, Ypk[:, 4], linestyle='-.', color='blue')
    plt.plot(t, Ypk[:, 14], linestyle='-.', color='orange')
    plt.plot(t, Ysp[:,4], linestyle='--', color='blue', label="Óleo esperado Poço 1")
    plt.plot(t, Ysp[:,14], linestyle='--', color='orange', label="Óleo esperado Poço 2")
    plt.ylabel("Vazão [kg/s]")
    plt.title("Vazão de Óleo Produzido")
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, Ymk[:, 6], label="Pressão na cabeça Poço 1")
    plt.plot(t, Ymk[:, 16], label="Pressão na cabeça Poço 2")
    plt.plot(t, Ypk[:, 6], linestyle='-.', color='blue')
    plt.plot(t, Ypk[:, 16], linestyle='-.', color='orange')
    plt.plot(t, Ysp[:, 6], linestyle='--', color='blue', label="Pressão de Cabeça Esperada Poço 1")
    plt.plot(t, Ysp[:, 16], linestyle='--', color='orange', label="Pressão de Cabeça Esperada Poço 2")
    plt.ylabel("Pressão [Pa]")
    plt.xlabel("Tempo [s]")
    plt.title("Pressões na Cabeça dos Poços")
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, Ymk[:, 8], label="Pbh Poço 1 (Pressão de Fundo)")
    plt.plot(t, Ymk[:, 18], label="Pbh Poço 2 (Pressão de Fundo)")
    plt.plot(t, Ypk[:, 8], linestyle='-.', color='blue')
    plt.plot(t, Ypk[:, 18], linestyle='-.', color='orange')
    plt.plot(t, Ysp[:, 8], linestyle='--', color='blue', label="Pressão de Fundo Esperada Poço 1")
    plt.plot(t, Ysp[:, 18], linestyle='--', color='orange', label="Pressão de Fundo Esperada")
    plt.ylabel("Pressão [Pa]")
    plt.xlabel("Tempo [s]")
    plt.title("Pressão de Fundo dos Poços")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "saida.png"))

with open('ENGG17 - Introdução à Elevação de Petróleo/results_NMPC.pkl', 'rb') as f:
    iter, Ymk, Ypk, Upk, dU, Ysp, Tempos = pickle.load(f)

plot_results(iter, Ymk, Ypk, Upk, dU, Ysp, Tempos)