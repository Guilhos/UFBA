import streamlit as st
from pages import circuit_rlc  

st.set_page_config(page_title="Simulador de Processos Contínuos", layout="wide")

# Inicializa o estado da página
if "page" not in st.session_state:
    st.session_state.page = "home"

# Página Home
def home():
    st.title("Simulador de Processos Contínuos")
    st.write("""
    Este aplicativo simula processos contínuos utilizando diferentes métodos de integração numérica.
    """)
    
    if st.button("Circuito RLC"):
        st.session_state.page = "circuit_rlc"

# Controle de fluxo
if st.session_state.page == "home":
    home()
elif st.session_state.page == "circuit_rlc":
    circuit_rlc.run()
