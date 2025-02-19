####### IMPORTAÇÃO DAS DEPENDÊNCIAS ########
import streamlit as st  # Biblioteca principal para criar aplicativos interativos
import pandas as pd  # Manipulação de dados
import plotly.express as px  # Gráficos interativos
import matplotlib.pyplot as plt  # Visualização de gráficos
import seaborn as sns  # Visualização avançada
import numpy as np  # Cálculos matemáticos
import plotly.graph_objects as go  # Gráficos interativos avançados
import gdown
import os
import pyarrow.parquet as pq
from datetime import datetime, timedelta  # Manipulação de datas
from scipy.interpolate import make_interp_spline  # Suavização de gráficos
from statsmodels.tsa.stattools import adfuller  # Teste ADF para estacionariedade
from statsmodels.tsa.seasonal import seasonal_decompose  # Decomposição de séries temporais
from statsmodels.tsa.arima.model import ARIMA  # Modelo ARIMA para previsão
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Modelo SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Métricas de avaliação
from prophet import Prophet  # Modelo Prophet do Facebook
from tensorflow.keras.models import Sequential  # Construção de modelos deep learning
from tensorflow.keras.layers import LSTM, Dense  # Camadas para redes neurais recorrentes
from sklearn.preprocessing import MinMaxScaler  # Normalização para machine learning

####### CONFIGURAÇÕES ########
st.set_page_config(layout="wide")  # Layout da página

# Sidebar para navegação
st.sidebar.title("Menu")
pagina_selecionada = st.sidebar.radio(
    'Aventure-se:',
    ["Brent: Histórico", "Estacionariedade, Tendências e Sazonalidades", 
     "Prophet", "LSTM", "Sobre o Desafio", "Sobre o Desenvolvedor"]
)

# Cores usadas nos gráficos
cor_primaria = "#3366CC"  # Azul refinado
cor_max = "#D62728"  # Vermelho forte
cor_min = "#2CA02C"  # Verde destacado
cor_destaque = "#FF7F0E"  # Cor quente para anos importantes

# Criar pasta "data" se não existir
if not os.path.exists("data"):
    os.makedirs("data")

####### FUNÇÕES ########

def baixar_e_carregar_parquet(file_id: str, output_file: str) -> pd.DataFrame:
    """
    Baixa o arquivo Parquet do Google Drive e carrega em um DataFrame.
    Se o arquivo já existir, apenas carrega sem baixar novamente.
    """
    if not os.path.exists(output_file):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_file, quiet=False)
            st.success("✅ Arquivo baixado com sucesso!")
        except Exception as e:
            st.error(f"🚨 Erro ao baixar o arquivo: {e}")
            return None

    try:
        df = pd.read_parquet(output_file, engine="pyarrow")
        st.write("✅ Dados carregados com sucesso!")
        return df
    except Exception as e:
        st.error(f"🚨 Erro ao carregar o arquivo Parquet: {e}")
        return None


def mostrar_grafico_temporal(df: pd.DataFrame, titulo: str, eixo_x: str, eixo_y: str):
    """Exibe um gráfico de linha interativo no Streamlit."""
    fig = px.line(df, x=eixo_x, y=eixo_y, title=titulo,
                  labels={eixo_x: 'Data', eixo_y: 'Preço (US$)'})
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço (US$)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def realizar_teste_adf(df: pd.DataFrame, coluna: str):
    """Executa o teste de Dickey-Fuller Aumentado (ADF) para verificar estacionariedade."""
    resultado_adf = adfuller(df[coluna])
    p_valor = resultado_adf[1]
    
    st.write(f"**Estatística ADF:** {resultado_adf[0]:.4f}")
    st.write(f"**p-valor:** {p_valor:.4f}")

    if p_valor < 0.05:
        st.success("A série é **estacionária** (p-valor < 0.05).")
    else:
        st.warning("A série **não é estacionária** (p-valor ≥ 0.05).")


####### CARREGAMENTO DOS DADOS ########
file_id = "1LjzB8BGdUroPRGKcHqOOLwCLbeRV_EUy"
output_file = "data/arquivo.parquet"
df = baixar_e_carregar_parquet(file_id, output_file)

if df is not None and "data" in df.columns:
    df["data"] = pd.to_datetime(df["data"])

####### INTERFACE DO APP ########
if pagina_selecionada == "Brent: Histórico":
    st.title('Brent: Histórico')
    
    if df is not None:
        anos_disponiveis = sorted(df['data'].dt.year.unique())
        col1, col2 = st.columns(2)
        with col1:
            ano_inicial = st.selectbox("Ano inicial", anos_disponiveis, index=0)
        with col2:
            ano_final = st.selectbox("Ano final", anos_disponiveis, index=len(anos_disponiveis)-1)

        df_filtrado = df[(df['data'].dt.year >= ano_inicial) & (df['data'].dt.year <= ano_final)]

        if not df_filtrado.empty:
            mostrar_grafico_temporal(df_filtrado, "Evolução do preço do Brent", "data", "preco")
        else:
            st.warning("Nenhum dado disponível para o período selecionado.")
    else:
        st.error("Erro ao carregar os dados.")

elif pagina_selecionada == "Estacionariedade, Tendências e Sazonalidades":
    st.title("Análise de Estacionariedade, Tendências e Sazonalidades")
    
    if df is not None:
        df_periodo = df[(df['data'] >= "2015-02-10") & (df['data'] <= "2025-02-10")]
        realizar_teste_adf(df_periodo, "preco")
    else:
        st.error("Erro ao carregar os dados.")

elif pagina_selecionada == "Prophet":
    st.title("Modelagem com Prophet")
    
    if df is not None:
        df_prophet = df[['data', 'preco']].rename(columns={'data': 'ds', 'preco': 'y'})
        modelo_prophet = Prophet()
        modelo_prophet.fit(df_prophet)
        
        futuro = modelo_prophet.make_future_dataframe(periods=365)
        previsoes = modelo_prophet.predict(futuro)
        
        mostrar_grafico_temporal(previsoes, "Previsão do Brent com Prophet", "ds", "yhat")
    else:
        st.error("Erro ao carregar os dados.")

elif pagina_selecionada == "Sobre o Desafio":
    st.title("Sobre o Desafio")
    st.write("📢 Explicação sobre o projeto da Pós-Graduação FIAP/Alura.")

elif pagina_selecionada == "Sobre o Desenvolvedor":
    st.title("Sobre o Desenvolvedor")
    st.write("👨‍💻 Desenvolvido por **Rogério Abramo Alves Pretti**, aluno da FIAP/Alura.")
