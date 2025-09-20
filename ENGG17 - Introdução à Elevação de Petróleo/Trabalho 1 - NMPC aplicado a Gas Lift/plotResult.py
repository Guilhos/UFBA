import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_results(iter, Ymk, Ypk, Upk, dU, Ysp, Tempos, dt):
    # Verifica se o diretório 'plots' existe, caso contrário, cria
    imagesPath = os.path.join(os.getcwd(), 'MPC-Freire/plots')
    os.makedirs(imagesPath, exist_ok=True)

    # Plot das entradas e pressões de fundo de poço

    t = np.arange(iter)*dt

    plt.figure(figsize=(10,5))
    plt.plot(t[:], Upk[:, 0], label="Injeção de Gás", color='blue')
    plt.plot(t[:], 5*np.ones_like(t), linestyle='--', color='red', label="Restrições")
    plt.plot(t[:], 0*np.ones_like(t), linestyle='--', color='red')
    plt.ylabel("Injeção de gás / kg/s", fontsize=24)
    plt.xlabel("Tempo / s", fontsize=24)
    plt.grid()
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(0.7, -0.2), ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "wgl1.png"))

    plt.figure(figsize=(10,5))
    plt.plot(t[:], Upk[:, 1], label="Injeção de Gás", color='orange')
    plt.plot(t[:], 5*np.ones_like(t), linestyle='--', color='red', label="Restrições")
    plt.plot(t[:], 0*np.ones_like(t), linestyle='--', color='red')
    plt.ylabel("Injeção de gás / kg/s", fontsize=24)
    plt.xlabel("Tempo / s", fontsize=24)
    plt.grid()
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(0.7, -0.2), ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "wgl2.png"))

    plt.figure(figsize=(10,5))
    plt.plot(t[:], Ypk[:, 0], label="Pressão de Fundo", color='blue')
    plt.plot(t[:], Ysp[:, 0], linestyle='--', color='black', label="Pressão de Fundo Esperada")
    plt.plot(t[:], 9.52e6*np.ones_like(t), linestyle='--', color='red', label="Restrições")
    plt.plot(t[:], 9.11e6*np.ones_like(t), linestyle='--', color='red')
    plt.ylabel("Pressão / Pa", fontsize=24)
    plt.xlabel("Tempo / s", fontsize=24)
    plt.grid()
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(0.85, -0.2), ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "pbh1.png"))

    plt.figure(figsize=(10,5))
    plt.plot(t[:], Ypk[:, 1], label="Pressão de Fundo", color='orange')
    plt.plot(t[:], Ysp[:, 1], linestyle='--', color='black', label="Pressão de Fundo Esperada")
    plt.plot(t[:], 9.74e6*np.ones_like(t), linestyle='--', color='red', label="Restrições")
    plt.plot(t[:], 9.46e6*np.ones_like(t), linestyle='--', color='red')
    plt.ylabel("Pressão / Pa", fontsize=24)
    plt.xlabel("Tempo / s", fontsize=24)
    plt.grid()
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(0.85, -0.2), ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, "pbh2.png"))

def calcular_ISDMV(sinal_controle):
    sinal_controle = np.array(sinal_controle).flatten()
    isdmv = np.sum(sinal_controle** 2)
    return isdmv

def calcular_ISE(referencia, saida):
    erro = np.array(referencia) - np.array(saida)
    ise = np.sum(erro**2)
    return ise


with open('MPC-Freire/results_NMPC.pkl', 'rb') as f:
    iter, Ymk, Ypk, Upk, dU, Ysp, Tempos, dt = pickle.load(f)

Ymk = np.array(Ymk)
Ypk = np.array(Ypk)
Upk = np.array(Upk)
dU = np.array(dU)
Ysp = np.array(Ysp)

plot_results(iter, Ymk, Ypk, Upk, dU, Ysp, Tempos, dt)

ISE_pbh1 = calcular_ISE(Ysp[:, 0], Ypk[:, 0])
ISE_pbh2 = calcular_ISE(Ysp[:, 1], Ypk[:, 1])
ISDMV_wgl1 = calcular_ISDMV(Upk[:, 0])
ISDMV_wgl2 = calcular_ISDMV(Upk[:, 1])

print(f"ISE pbh 1: {ISE_pbh1}")
print(f"ISE pbh 2: {ISE_pbh2}")
print(f"ISDMV wgl 1: {ISDMV_wgl1}")
print(f"ISDMV wgl 2: {ISDMV_wgl2}")