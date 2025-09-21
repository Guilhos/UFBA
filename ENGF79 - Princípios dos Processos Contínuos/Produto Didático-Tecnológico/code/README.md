projeto_experimentos/
├─ app.py                  # Arquivo principal do Streamlit (menu inicial)
├─ pages/                  # Páginas individuais do app (cada experimento)
│   ├─ 1_Tanque_de_Nivel.py
│   ├─ 2_Massa_Mola.py
│   ├─ 3_Circuito_RC.py
│   ├─ 4_Identificacao.py
│   └─ ...
├─ models/                 # Equações/matemática dos sistemas
│   ├─ tank.py
│   ├─ mass_spring.py
│   ├─ circuit_rc.py
│   └─ cstr.py
├─ sims/                   # Rotinas genéricas de simulação
│   ├─ integrators.py      # Euler, RK4, solve_ivp (Scipy)
│   └─ inputs.py           # Sinais de entrada: degrau, senóide, PRBS
├─ analysis/               # Ferramentas de análise
│   ├─ linearize.py        # Linearização simbólica
│   ├─ transfer_function.py
│   └─ identification.py   # LS/ARX simples
├─ utils/                  # Funções auxiliares (gráficos, formatação)
│   └─ plotting.py
├─ requirements.txt        # Dependências (streamlit, numpy, scipy, matplotlib...)
└─ README.md               # Documentação explicando como usar
