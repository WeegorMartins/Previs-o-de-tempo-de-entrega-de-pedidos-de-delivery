import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregar modelo e colunas
modelo = joblib.load('modelo_xgboost_entrega.pkl')
colunas_modelo = joblib.load('colunas_modelo.pkl')

st.set_page_config(page_title="Previsão de Entrega - iFood", layout="centered")
st.title("🍽️ Previsão de Tempo de Entrega")
st.write("Preencha as informações abaixo para prever o tempo estimado de entrega e verificar risco de SLA.")

# Campos do form
with st.form("formulario"):
    cidade = st.selectbox("Cidade", ['Metropolitian', 'Urban', 'Semi-Urban'])
    clima = st.selectbox("Condição Climática", ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Fog'])
    transito = st.selectbox("Trânsito", ['Low', 'Medium', 'High', 'Jam'])
    tipo_pedido = st.selectbox("Tipo de Pedido", ['Snack', 'Meal', 'Drinks', 'Buffet'])
    tipo_veiculo = st.selectbox("Tipo de Veículo", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
    idade = st.slider("Idade do Entregador", 18, 60, 30)
    avaliacao = st.slider("Avaliação do Entregador", 1.0, 5.0, 4.5, 0.1)
    cond_veiculo = st.selectbox("Condição do Veículo", [0, 1, 2, 3])
    multiplas_entregas = st.selectbox("Quantas entregas simultâneas?", [0, 1, 2, 3])
    festival = st.selectbox("É um festival?", ['Yes', 'No'])
    tempo_espera = st.number_input("Tempo de espera até retirada (min)", value=5.0)
    distancia = st.number_input("Distância até o cliente (km)", value=3.0)
    enviar = st.form_submit_button("Prever")

if enviar:
    entrada_dict = {
        'Delivery_person_Age': idade,
        'Delivery_person_Ratings': avaliacao,
        'Vehicle_condition': cond_veiculo,
        'multiple_deliveries': multiplas_entregas,
        'waiting_time': tempo_espera,
        'distance_km': distancia,
        'Festival_' + festival: 1,
        'City_' + cidade: 1,
        'Weatherconditions_' + clima: 1,
        'Road_traffic_density_' + transito: 1,
        'Type_of_order_' + tipo_pedido: 1,
        'Type_of_vehicle_' + tipo_veiculo: 1,
    }

    # Construir DataFrame com todas as colunas usadas no treino
    dados_input = pd.DataFrame(columns=colunas_modelo)
    dados_input.loc[0] = 0
    for k, v in entrada_dict.items():
        if k in dados_input.columns:
            dados_input.at[0, k] = v

    # Previsão
    tempo_estimado = modelo.predict(dados_input)[0]

    # Avaliação de SLA
    if tempo_estimado <= 30:
        st.success(f"🚀 Tempo estimado: {tempo_estimado:.1f} minutos. SLA OK!")
    else:
        st.error(f"⚠️ Tempo estimado: {tempo_estimado:.1f} minutos. SLA em risco!")
