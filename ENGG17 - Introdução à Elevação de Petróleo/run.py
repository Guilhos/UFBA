from nmpc import NMPC
import pickle

# Lista de saídas (outputs):
        #  0: wiv1     - Vazão de gás injetado no Poço 1
        #  1: wro1     - Vazão de óleo do Poço 1
        #  2: wpc1     - Vazão de líquido no choke do Poço 1
        #  3: wpg1     - Vazão de gás do Poço 1 que vai para o riser
        #  4: wpo1     - Vazão de óleo do Poço 1 que vai para o riser
        #  5: Pai1     - Pressão no anular de injeção do Poço 1
        #  6: Pwh1     - Pressão na cabeça do Poço 1
        #  7: Pwi1     - Pressão de entrada do poço 1 (em profundidade)
        #  8: Pbh1     - Pressão de fundo do Poço 1
        #  9: wrg1     - Vazão de gás de reservatório do Poço 1
        # 10: wiv2     - Vazão de gás injetado no Poço 2
        # 11: wro2     - Vazão de óleo do Poço 2
        # 12: wpc2     - Vazão de líquido no choke do Poço 2
        # 13: wpg2     - Vazão de gás do Poço 2 que vai para o riser
        # 14: wpo2     - Vazão de óleo do Poço 2 que vai para o riser
        # 15: Pai2     - Pressão no anular de injeção do Poço 2
        # 16: Pwh2     - Pressão na cabeça do Poço 2
        # 17: Pwi2     - Pressão de entrada do poço 2 (em profundidade)
        # 18: Pbh2     - Pressão de fundo do Poço 2
        # 19: wrg2     - Vazão de gás de reservatório do Poço 2
        # 20: wtg      - Vazão total de gás no riser
        # 21: wto      - Vazão total de óleo no riser
        # 22: Prh      - Pressão no topo do riser
        # 23: Pm       - Pressão média no riser
        # 24: ro_r     - Densidade média do fluido no riser
        # 25: wrh      - Vazão total no riser (gás + óleo)

Q = [1e-2,1e-2]
R = [1,1]

p, m, steps = 10, 3, 3
NMPC = NMPC(p, m, steps, 2, 8, 2, Q, R, 10, 1, 160)
iter, Ymk, Ypk, Upk, dU, Ysp, Tempos = NMPC.run()

# Salvando os resultados em um arquivo pickle
with open('ENGG17 - Introdução à Elevação de Petróleo/results_NMPC.pkl', 'wb') as f:
    pickle.dump((
        iter,
        Ymk,
        Ypk,
        Upk,
        dU,
        Ysp,
        Tempos
    ), f)