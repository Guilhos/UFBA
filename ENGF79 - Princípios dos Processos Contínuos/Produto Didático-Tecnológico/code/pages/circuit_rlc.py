import streamlit as st

def run():
    st.title("Método de Integração 1")
    st.write("Aqui está a simulação do método 1.")

    # Botão para voltar para home
    if st.button("Voltar para Home"):
        st.session_state.page = "home"
