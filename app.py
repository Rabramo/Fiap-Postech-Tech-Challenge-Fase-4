import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import make_interp_spline

## configuração de estilo global
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: 'Lato', sans-serif;
            color: #333333;
        }
        .stTitle { font-size: 20px !important; font-weight: bold; color: #3366CC; }
        .stHeader { font-size: 16px !important; font-weight: bold; color: #FF7F0E; }
        .stSubheader { font-size: 14px !important; font-weight: bold; color: #3366CC; }
        .stMarkdown { font-size: 14px; line-height: 1.6; }
        .stButton>button { background-color: #FF7F0E !important; color: white !important; font-size: 16px !important; font-weight: bold !important; border-radius: 8px !important; }
        .stDataFrame { font-size: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True
)
# Configuração do estilo dos gráficos
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.alpha": 0.3
})

# Cores 
cor_primaria = "#3366CC"  # Azul refinado
cor_max = "#D62728"  # Vermelho impactante
cor_min = "#2CA02C"  # Verde destacado
cor_destaque = "#FF7F0E"  # Cor quente para anos importantes

# Função para obter os preços do Brent do yfinance
df = pd.read_csv('https://raw.githubusercontent.com/Rabramo/Fiap-Postech-Tech-Challenge-Fase-4/refs/heads/main/ipeadata%5B17-02-2025-07-20%5D.csv', sep=',', decimal=',')

# tratando as colunas
# renomear a coluna do preço
#excluir linhas que tenham Nan na coluna Preço - petróleo bruto - Brent (FOB) - US$
df.dropna(subset=['Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366'], inplace=True)

df.rename(columns={'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366':
                   'preco_brent_fob_US$'}, inplace=True)

# Converter a coluna 'Data' para o tipo datetime e colocar todas as letras me minúscula
df['data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

# excluir colunas Data e  Unnamed: 2
df.drop(columns=['Data', 'Unnamed: 2'], inplace=True)

# Reordenar as colunas
df = df[['data', 'preco_brent_fob_US$']]

#resetar o índice para garantir as alterações esteja indexadas
df.reset_index(drop=True, inplace=True)

#Definição de limites de datas
#ano_atual = datetime.today().year
#ano_min = 1997
#ano_max = ano_atual


# Título da páginastrea
st.title('Brent: Acompanhe o Mercado')
'''
O Brent é um dos principais benchmarks do mercado global de petróleo e sua volatilidade 
o torna uma grande oportunidade para investidores que buscam ganhos rápidos. Oscilando 
conforme fatores como tensões geopolíticas, crises econômicas, oferta e demanda, além da 
especulação financeira, seu preço pode sofrer variações significativas em curtos períodos. 
Essa característica atrai traders e fundos de investimento que aproveitam essas flutuações 
para lucrar, seja por meio de contratos futuros, opções ou ETFs ligados ao petróleo. 
No entanto, essa mesma volatilidade também exige estratégias bem planejadas e uma gestão 
de risco eficiente para evitar perdas.

'''
st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço
# Seletor de intervalo de datas
st.header('Defina o intervalo, veja a tendência')

#st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço

col1, col2 = st.columns(2)

₢

# Garantindo que 'data' não tenha timezone
df["data"] = df["data"].dt.tz_localize(None)

# Filtrar os dados
df_filtered = df[(df['data'] >= pd.to_datetime(start_year)) & (df['data'] <= pd.to_datetime(end_year))]

# Adicionar coluna de ano
df_filtered['ano'] = df_filtered['data'].dt.year

# Filtrar anos destacados de acordo com a seleção
destaque_anos = {2008, 2014, 2020, 2022} & set(df_filtered['ano'].unique())

st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço


# Gráfico de Boxplot com destaque nos anos selecionados
#st.subheader('Preço do barril do petróleo Brent por Ano')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='ano', y='preco_brent_fob_US$', data=df_filtered, ax=ax, width=0.6, fliersize=3, palette=[
    cor_destaque if ano in destaque_anos else cor_primaria for ano in df_filtered['ano'].unique()
])

# Rotacionar os rótulos do eixo X para 45 graus
plt.xticks(rotation=45)

ax.set_xlabel(" ")
ax.set_ylabel(" ")
ax.set_title("Preço do barril de Brent (US$)", fontsize=18, fontweight="bold")
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 

# Gráfico de Linha - Evolução do preço
#st.subheader('Evolução do Preço do Petróleo Brent')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_filtered['data'], df_filtered['preco_brent_fob_US$'], color=cor_primaria, linewidth=2, alpha=0.9)
ax.scatter(df_filtered['data'].iloc[df_filtered['preco_brent_fob_US$'].idxmax()], df_filtered['preco_brent_fob_US$'].max(), color=cor_max, s=100, label="Máximo")
ax.scatter(df_filtered['data'].iloc[df_filtered['preco_brent_fob_US$'].idxmin()], df_filtered['preco_brent_fob_US$'].min(), color=cor_min, s=100, label="Mínimo")
ax.set_xlabel("")
ax.set_ylabel("Preço (US$)", fontsize=16, fontweight="bold")
ax.set_title("📊 Evolução do Preço do Petróleo Brent", fontsize=18, fontweight="bold")
ax.legend()
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 

# Gráfico de distribuição de frequências
#st.subheader('📊 Distribuição dos Preços do Petróleo Brent')
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(df_filtered['preco_brent_fob_US$'], kde=True, color=cor_primaria, bins=25, ax=ax)
ax.set_xlabel("Preço (US$)", fontsize=16, fontweight="bold")
ax.set_ylabel("Frequência", fontsize=16, fontweight="bold")
ax.set_title("📈 Distribuição dos Preços do Petróleo Brent", fontsize=18, fontweight="bold")
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 

if st.button('Sobre o Brent e fonte de dados utilizada'):
        st.markdown('''
        **Origem dos Dados:** Os dados utilizados nesta análise foram obtidos através da biblioteca `yfinance`, 
        que permite o acesso a informações financeiras diretamente do Yahoo Finance. São dados referentes ao 
        ticker "BZ=F" é utilizado para representar os contratos futuros do petróleo Brent no Yahoo Finance. Esses 
        contratos são negociados na Intercontinental Exchange (ICE) e são amplamente utilizados como referência 
        para o preço do petróleo bruto no mercado global.

        O barril de Brent é uma unidade de medida de volume utilizada na indústria petrolífera. 
        Um barril de petróleo equivale a 42 galões americanos, ou aproximadamente 159 litros.

        O Brent é um tipo de petróleo extraído do Mar do Norte, especificamente das áreas de Brent e Ninian.É considerado 
        um petróleo leve (light) e doce (sweet) devido ao seu baixo teor de enxofre, o que facilita o processo de refino.
        Foi adotado como benchmark para a precificação de outros tipos de petróleo ao redor do mundo. 
        Outros benchmarks incluem o West Texas Intermediate (WTI) e o Dubai Crude.
                          
        '''
        )
        st.button("❌ Fechar", key='fechar')

# Criar modais para anos em destaque
chamada = {
    2008: "Crise financeira global e pico histórico do petróleo.",
    2014: "Aumento da produção de xisto nos EUA e queda nos preços.",
    2020: "Impacto da pandemia da COVID-19 e colapso na demanda.",
    2022: "Efeitos da guerra na Ucrânia e sanções econômicas."
}

analise = {

    2008: """
**2008: Crise Financeira Global**

Em 2008, o mercado de petróleo experimentou uma volatilidade extrema devido à crise financeira global. Antes da crise, 
os preços atingiram níveis recordes devido à alta demanda e especulação. Em julho o barril alcançou o maior valor nominal até hoje registrado: U$ 147,50.

Com o colapso financeiro, a demanda por petróleo caiu drasticamente, levando a uma queda acentuada nos preços. Em dezembro de 2008, o barril chegou a ser cotado a U$ 33,87.

""", 
    2014: """
**2014: Aumento da Produção de Xisto nos EUA e Queda nos Preços**

Em 2014, o mercado enfrentou uma superabundância de oferta devido ao aumento da produção de petróleo de xisto nos Estados Unidos e à decisão 
da OPEC de não reduzir a produção. Simultaneamente, a desaceleração econômica na China e em outras economias emergentes reduziu a demanda por petróleo, 
resultando em uma queda significativa nos preços.

O maior valor foi US$ 115,19 por barril em junho de 2014. O menor, US$ 53,27 em dezembro de 2014.
""",
    2020: """
    **2020: Impacto da Pandemia da COVID-19 e Colapso na Demanda**

    A pandemia de COVID-19 levou a uma queda drástica na demanda por petróleo devido a lockdowns e redução de atividades econômicas. 
    Além disso, em março de 2020, a Rússia e a Arábia Saudita entraram em uma guerra de preços, aumentando a produção e exacerbando a queda nos preços.

    Vimos o barril ser cotado a US$ 11,14 em abril de 2020, após ter atingido US$ 65,65 em janeiro do mesmo ano.
    """,
    2022: """   
    **Recuperação Pós-Pandemia e Geopolítica**

    Em 2022, a recuperação econômica pós-pandemia aumentou a demanda por petróleo. Além disso, a invasão da Ucrânia pela Rússia em fevereiro de 2022 
    gerou incertezas no mercado, afetando a oferta e a demanda e contribuindo para a volatilidade dos preços.

    O barril de petróleo Brent atingiu US$ 139,13 em março de 2022, após ter iniciado o ano cotado a US$ 79,25. Em dezembro, foi atingiu o menor valor: US$ 77,78.

    """
}


#st.subheader("📌 Explicação dos Anos em Destaque")
for ano in sorted(destaque_anos):
    if st.button(f"{ano} - {chamada[ano]}"):
        st.markdown(f"{analise[ano]}")
        st.button("❌ Fechar", key=f"fechar_{ano}")



