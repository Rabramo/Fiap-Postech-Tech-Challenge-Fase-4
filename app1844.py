####### IMPORTAÇÃO DAS DEPENDÊNCIAS ########
import streamlit as st  # Biblioteca principal para criar aplicativos interativos
import pandas as pd  # Manipulação de dados em tabelas DataFrame
import plotly.express as px  # Importação da biblioteca Plotly Express para gráficos interativos
import locale  # Configuração de localidade (moeda, número, data, etc.)
import matplotlib.pyplot as plt  # Biblioteca para visualização de gráficos
import seaborn as sns  # Biblioteca para visualização de dados avançada
import numpy as np  # Cálculos matemáticos avançados e manipulação de arrays
import plotly.graph_objects as go  # Biblioteca para criação de visualizações interativas, como gráficos de linha, barras e dispersão
import requests # para fazer requisições comom baixar um arquivo por ex.
import io
from datetime import datetime, timedelta  # Manipulação de datas e períodos de tempo
from scipy.interpolate import make_interp_spline  # Interpolação para suavizar gráficos, útil para visualizações mais suaves
from statsmodels.tsa.stattools import adfuller  # Teste de Dickey-Fuller aumentado (ADF) para verificar estacionariedade de séries temporais
from statsmodels.tsa.seasonal import seasonal_decompose  # Decomposição de séries temporais em tendência, sazonalidade e resíduo
from statsmodels.tsa.arima.model import ARIMA  # Modelo ARIMA (AutoRegressive Integrated Moving Average) para previsão de séries temporais
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Modelo SARIMAX (ARIMA sazonal) para séries temporais com sazonalidade e variáveis exógenas
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Métricas para avaliação de modelos: erro médio absoluto (MAE) e erro médio quadrático (MSE)
from prophet import Prophet  # Modelo de previsão de séries temporais desenvolvido pelo Facebook, baseado em decomposição aditiva
from tensorflow.keras.models import Sequential  # API para construção de modelos de deep learning sequenciais no TensorFlow/Keras
from tensorflow.keras.layers import LSTM, Dense  # LSTM: camada de rede neural recorrente para séries temporais; Dense: camada totalmente conectada
from sklearn.preprocessing import MinMaxScaler  # Normalização de dados para modelos de machine learning, escalando valores para um intervalo específico


####### CONFIGURAÇÕES ########
# Configuração do Streamlit
# Definir a largura da página do app para 'auto' que Dinamicamente ajusta entre "centered" e "wide", dependendo do tamanho da tela do usuário.
st.set_page_config(layout="wide")

# Sidebar com menu de navegação
st.sidebar.title("Menu")
pagina_selecionada = st.sidebar.radio('Aventure-se:',["Brent: Histórico", "Estacionariedade, Tendências e Sazonalidades", 
                                                        "Prophet", "LSTM", "Video Explicativo", "Sobre o Desafio", "Sobre o Desenvolvedor"])

# Definir estilo do app
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: 'Lato', sans-serif;
            color: #333333;
        }
        .stTitle { font-size: 18px !important; font-weight: bold; color: #3366CC; }
        .stHeader { font-size: 14px !important; font-weight: bold; color: #3366CC; }
        .stSubheader { font-size: 14px !important; font-weight: bold; color: #3366CC; }
        .stMarkdown { font-size: 14px; line-height: 1.6; }
        .stButton>button { background-color: #FF7F0E !important; color: white !important; font-size: 16px !important; font-weight: bold !important; border-radius: 8px !important; }
        .stDataFrame { font-size: 14px !important; }
        .main .block-container {
            max-width: 1400px;  /* Ajuste esse valor conforme necessário */ }
        .container {
            max-width: 33%;  /* Define que ocupa 1/3 do frame */
            margin: auto;  /* Centraliza no meio da tela */
        }   
        div[data-testid="stDateInput"] input {
            width: 120px !important;  /* Ajuste a largura */
            font-size: 14px !important; /* Ajuste o tamanho da fonte */
            padding: 7px !important; /* Ajuste o espaçamento interno */
        }
        
        div[data-baseweb="select"] {
            width: 120px !important;  /* largura */
            color: white !important;              /* Cor da fonte */
            font-size: 16px !important;           /* Tamanho da fonte */
            #border-radius: 8px !important;        /* Borda arredondada */
            padding: 5px !important;              /* Espaçamento interno */
        }
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

####### CARREGAR DADOS ########
# Criando DF a partir do parquet com dados do Brent do Ipea (https://www.ipeadata.gov.br/Default.aspx), usando r, raw string para evitar problemas com barras
# Função para carregar dados
@st.cache_data
def carregar_dados():
    # URL do arquivo Parquet no GitHub usando link RAW
    url = "https://raw.githubusercontent.com/Rabramo/Fiap-Postech-Tech-Challenge-Fase-4/main/ipea_brent_20250217.parquet"
    try:
        # Baixar o arquivo como stream de bytes
        response = requests.get(url)
        response.raise_for_status()  # Verifica se o download foi bem-sucedido

        # Carregar o arquivo Parquet a partir dos bytes
        df = pd.read_parquet(io.BytesIO(response.content), engine="pyarrow")

        # Criar a coluna 'ano' extraindo do campo de data
        df['ano'] = pd.to_datetime(df['data']).dt.year

        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar os dados: {e}")
        return None

# Chamar a função para carregar os dados
df = carregar_dados()
# Exibir os dados no Streamlit
# if df is not None:
st.success("✅ Dados carregados com sucesso!")
#   st.dataframe(df)
# Recuperando datas selecionadas na página Brent
if "ano_inicial" in st.session_state and "ano_final" in st.session_state:
    data_inicio = pd.to_datetime(f"{st.session_state['ano_inicial']}-01-01")
    data_fim = pd.to_datetime(f"{st.session_state['ano_final']}-12-31")
else:
    data_inicio = pd.to_datetime("2015-02-10")
    data_fim = pd.to_datetime("2025-02-10")

# Controle de exibição de páginas
st.session_state.setdefault("pagina_anterior", None)


#%% Brent: Converta Volatilidade em Ganho
# Conteúdo da Brent: Converta Volatilidade em Ganho
if pagina_selecionada == "Brent: Histórico":
    st.title('Brent: Domine a Volatilidade e Converta Oscilações em Lucros')
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

    # Criando duas colunas para dividir os botões principais lado a lado
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1]) 

    # Botões principais (lado a lado)
    with col1:
        exibir_brent = st.button("Saiba mais sobre o Brent", key="btn_brent")

    with col2:
        exibir_fonte = st.button("Fonte de dados", key="btn_fonte")

    # Exibir informações quando os botões forem pressionados
    if exibir_brent:
        st.markdown('''
        O barril de Brent é uma unidade de medida de volume utilizada na indústria petrolífera. 
        Um barril de petróleo equivale a 42 galões americanos, ou aproximadamente 159 litros.

        O Brent é um tipo de petróleo extraído do Mar do Norte, especificamente das áreas de Brent e Ninian. 
        É considerado um petróleo leve (light) e doce (sweet) devido ao seu baixo teor de enxofre, o que facilita o processo de refino.
        Foi adotado como benchmark para a precificação de outros tipos de petróleo ao redor do mundo. 
        Outros benchmarks incluem o West Texas Intermediate (WTI) e o Dubai Crude.                          
        ''')

        # Botão de fechar
        if st.button("Fechar", key="close_brent"):
            st.experimental_rerun()  # Recarrega a página para ocultar o conteúdo

    if exibir_fonte:
        st.markdown('''
        Os dados utilizados neste aplicativo foram obtidos do Ipeadata, um portal de dados econômicos do Instituto de Pesquisa
        Econômica Aplicada (Ipea). Originalmente os dados são da Energy Information Administration (EIA) dos Estados Unidos. 
        O preço é o FOB - Free on Board Basis (preço livre a bordo), que não inclui custos de transporte e seguro, e spot, utilizado
        nas negociações à vista, em dólares americanos, por barril.
        Mais informações em http://www.ipeadata.gov.br/.                          
        ''')

        # Botão de fechar
        if st.button("Fechar", key="close_fonte"):
            st.experimental_rerun()  # Recarrega a página para ocultar o conteúdo

    # Seletor de intervalo de datas
    st.header('Escolha o intervalo e comece sua aventura pelo mundo do Brent')

    anos = list(range(1988, 2026))  # Lista de anos disponíveis

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Criando colunas para alinhar melhor, 0.1 e 1 são proporções de largurapara deixa o select lado a lado

    with col1:
        ano_inicial = st.selectbox("", anos, index=0, key="ano_inicial", label_visibility="collapsed")  # Atribuindo chave única

    with col2:
        ano_final = st.selectbox("", anos, index=len(anos)-1, key="ano_final", label_visibility="collapsed")  # Atribuindo chave única

    # Converter anos selecionados para datetime
    ano_inicial = pd.to_datetime(f"{ano_inicial}-01-01")
    ano_final = pd.to_datetime(f"{ano_final}-12-31")

    df["data"] = pd.to_datetime(df["data"], errors='coerce')  
    # Filtrar o DataFrame pelo intervalo de anos
    df_filtrado = df[(df["data"] >= ano_inicial) & (df["data"] <= ano_final)]

    # Garantir que o DataFrame não está vazio antes de continuar
    if df_filtrado.empty:
        st.warning("Nenhum dado disponível para o intervalo selecionado.")
    else:
        # Pegando os valores máximo e mínimo
        max_idx = df_filtrado['preco'].idxmax()
        min_idx = df_filtrado['preco'].idxmin()

        # Resetando o índice para garantir que idxmax() e idxmin() são acessíveis com iloc[]
        df_filtrado = df_filtrado.reset_index(drop=True)

    # Criar coluna de ano para destaque
    df_filtrado['ano'] = df_filtrado['data'].dt.year
    destaque_anos = {1990, 1998, 1999, 2008, 2014, 2020, 2022} & set(df_filtrado['ano'].unique())

    # Verificação se há pelo menos um ano relevante no período selecionado
    if not destaque_anos:
        st.warning("Para o período selecionado não consta nenhuma análise, altere para incluir mais anos.")
    else:
        for ano in sorted(destaque_anos):
            if ano == 1990:
                descricao = "Invasão do Kuwait pelo Iraque e Guerra do Golfo."
                texto = """
                    O mercado de petróleo entrou em 1990 com uma relativa estabilidade após a crise do petróleo dos anos 1980, 
            com preços girando em torno de 18 a 22 dólares por barril. No entanto, a situação mudou drasticamente com a invasão do Kuwait 
            pelo Iraque em 2 de agosto de 1990, desencadeando a Guerra do Golfo e a disparada no preço do petróleo. O preço do Brent dobrou em poucas semanas, 
            ultrapassando 40 dólares por barril em outubro, o maior valor desde a crise do petróleo de 1979.

            O motivo da instabilidade com aumento do valor do Brent foi o receio de escassez no mercado, já que o Kuwait era um grande produtor de petróleo, e o Iraque 
            possuía uma das maiores reservas do mundo.
            
            Porém, já em dezembro de 1990, com a vitória da coalizão liderada pelos Estados Unidos na Guerra do Golfo, o preço do petróleo começou a cair, fechando o 
            ano com o barril custando 28,35 dólares.
                """
            elif ano == 1998:
                descricao = "Crise financeira asiática e queda nos preços do petróleo."
                texto = """
                A Crise Financeira Asiática, que começou em 1997, teve um impacto devastador nos mercados globais, e seus efeitos se intensificaram em 1998. 
        A crise começou na Tailândia, espalhou-se rapidamente para Indonésia, Coreia do Sul, Malásia e outras economias asiáticas, causando falências em massa, 
        desvalorização de moedas e retração do crescimento econômico.

        A Ásia é um grande consumidor de petróleo, e a crise reduziu drasticamente a demanda. Empresas faliram, governos entraram em colapso e o crescimento 
        econômico na região caiu para perto de zero. Investidores fugiram dos mercados emergentes, e a especulação aumentou a incerteza global.

        Com menos consumo de petróleo pela Ásia, o excesso de oferta ficou ainda mais evidente, pressionando os preços para baixo. 1998 foi marcado por uma das 
        maiores quedas nos preços do petróleo Brent da década. O petróleo, que vinha sendo negociado acima de 20 dólares por barril em anos anteriores, 
        chegou a cair para menos de 10 dólares, atingindo um dos níveis mais baixos da história moderna.
            """
            elif ano == 1999:
                descricao = "Crescimento econômico global e aumento da demanda."
                texto = """
            O início de 1999 foi de preços baixos, no rescaldo da forte desvalorização em 1998, com o barril sendo negociado próximo de 10-12 dólares.
        A OPEP, junto com outros produtores, decidiu cortar a produção para reequilibrar o mercado. 
        
        Com a economia global começando a se
        recuperar, aumentando a demanda para uma oferta artificialmente limitada pelos produtores, o preço do Brent começou a subir. Fechou 1999 custando 
        cerca de 25 doláres, mais do que o dobro dos valores do início do ano.
            """
            elif ano == 2008:
                descricao = "Crise financeira global e pico histórico do petróleo."
                texto = """
           Em 2008, o mercado de petróleo experimentou uma volatilidade extrema devido à crise financeira global. Antes da crise, 
    os preços atingiram níveis recordes devido à alta demanda e especulação. Em julho o barril alcançou o maior valor nominal até hoje registrado: U$ 143,95.

    Com o colapso financeiro, a demanda por petróleo caiu drasticamente, levando a uma queda acentuada nos preços. Em dezembro de 2008, o barril chegou a ser cotado a U$ 33,87.
            """
            elif ano == 2014:
                descricao = "Aumento da produção de xisto nos EUA e queda nos preços."
                texto = """
            Em 2014, o mercado enfrentou uma superabundância de oferta devido ao aumento da produção de petróleo de xisto nos Estados Unidos e à decisão 
    da OPEC de não reduzir a produção. Simultaneamente, a desaceleração econômica na China e em outras economias emergentes reduziu a demanda por petróleo, 
    resultando em uma queda significativa nos preços.

    O maior valor foi U$ 115,19 por barril em junho de 2014. O menor, U$ 53,27 em dezembro de 2014.
            """
            elif ano == 2020:
                descricao = "Impacto da pandemia da COVID-19 e colapso na demanda."
                texto = """
            A pandemia de COVID-19 levou a uma queda drástica na demanda por petróleo devido a lockdowns e redução de atividades econômicas. 
            Além disso, em março de 2020, a Rússia e a Arábia Saudita entraram em uma guerra de preços, aumentando a produção e exacerbando a queda nos preços.
            Vimos o barril ser cotado a U$  11,14 
            em abril de 2020, após ter atingido U$  65,65 em janeiro do mesmo ano.
            """
            elif ano == 2022:
                descricao = "Efeitos da guerra na Ucrânia e sanções econômicas."
                texto = """
                Em 2022, a recuperação econômica pós-pandemia aumentou a demanda por petróleo. Além disso, a invasão da Ucrânia pela Rússia em fevereiro de 2022 
                gerou incertezas no mercado, afetando a oferta e a demanda e contribuindo para a volatilidade dos preços.

                O barril de petróleo Brent atingiu 139,13 em março de 2022, após ter iniciado o ano cotado a 79,25. Em dezembro, 
                foi atingido o menor valor: 77,78 dólares.
            """
            else:
                descricao = "Descrição não disponível."
                texto = "Sem informações detalhadas para este ano."

            # 🔹 Correção da indentação: o botão deve estar dentro do loop para cada ano
            if st.button(f"{ano} - {descricao}"):
                st.markdown(texto)
                st.button("❌ Fechar", key=f"fechar_{ano}")

    st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 
    # Gráfico de Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = [cor_destaque if ano in destaque_anos else cor_primaria for ano in df_filtrado['ano'].unique()]
    sns.boxplot(x='ano', y='preco', data=df_filtrado, ax=ax, width=0.6, fliersize=3, palette=cmap)
    plt.xticks(rotation=45)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    ax.set_title("Boxplot do preço do barril de Brent (US$)", fontsize=18, fontweight="bold")
    st.pyplot(fig)

    col1, col2, col3, col4 = st.columns([0.3, 0.5, 0.8, 1.5])

    # Botões principais (lado a lado)
    with col1:
        exibir_o_que_e = st.button('O que é um Boxplot?', key="btn_o_que_e")

    with col2:
        exibir_interpretar = st.button('Como interpretar um Boxplot?', key="btn_interpretar")

    with col3:
        exibir_porque_usar = st.button('Por que usar um Boxplot para analisar o Brent?', key="btn_porque_usar")

    if exibir_o_que_e:
            st.markdown('''
            O Boxplot (ou diagrama de caixa) é um gráfico estatístico usado para visualizar a distribuição de um conjunto de dados e identificar outliers, dispersão e a mediana de uma variável.

            Ele é composto por cinco estatísticas principais:

            *Mínimo (sem contar outliers)*
            
            *Primeiro quartil (Q1 - 25%)*
            
            *Mediana (Q2 - 50%)*
            
            *Terceiro quartil (Q3 - 75%)*
            
            *Máximo (sem contar outliers)*
            
            Além disso, ele destaca outliers, que são valores anormalmente altos ou baixos.                  
            '''
            )
            st.button("x")

    if exibir_interpretar:
            st.markdown('''
        A caixa representa 50% dos dados (entre Q1 e Q3).
        
        A linha dentro da caixa é a mediana (valor central da distribuição).
        
        Os "bigodes" mostram a dispersão dos dados sem outliers.
        
        Os pontos fora dos bigodes são outliers, ou seja, valores muito diferentes da maioria dos dados.
        
        Exemplo de interpretação:
        
        Se a caixa for estreita, os preços do Brent têm baixa volatilidade.
        
        Se a caixa for larga, há alta volatilidade.
        
        Se houver muitos outliers, significa que há períodos de preços extremos (picos ou quedas bruscas).
        
        Se a mediana estiver mais próxima do Q1 ou do Q3, a distribuição dos preços é assimétrica.            
            '''
            )
            st.button("x")

    if exibir_porque_usar:
            st.markdown('''
                 O preço do Petróleo Brent é altamente volátil e pode sofrer grandes variações devido a fatores como:
                 
            - Choques econômicos (crises financeiras, pandemias, guerras).

            - Decisões da OPEP (cortes ou aumentos na produção).

            - Oscilações no dólar.

            - Mudanças na demanda global.
            
            - Tensões geopolíticas.
            

            Com um Boxplot, você pode: 
            
            - Identificar períodos de alta volatilidade: Se os bigodes são longos ou há muitos outliers, indica que o Brent teve flutuações bruscas.
            - Comparar diferentes períodos: Podemos analisar se a volatilidade do Brent está aumentando ou diminuindo ao longo do tempo.
            - Verificar anomalias: Outliers indicam períodos em que o petróleo atingiu valores extremos.

            O Boxplot é uma ferramenta poderosa para visualizar rapidamente a distribuição dos preços do Brent, 
            identificando tendências de volatilidade e anomalias. 
            Ele ajuda traders, investidores e analistas a entender melhor o comportamento do mercado e tomar decisões estratégicas.
       
            '''
            )
            st.button("x")

    st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 

    # Gráfico de Linha - Evolução do preço
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pegando os valores máximo e mínimo
    max_idx = df_filtrado['preco'].idxmax()
    min_idx = df_filtrado['preco'].idxmin()
    max_val = df_filtrado['preco'].max()
    min_val = df_filtrado['preco'].min()

    # Criando pontos suavizados para interpolação cúbica
    x_original = df_filtrado['data'].astype(np.int64) // 10**9  # Convertendo data para timestamp (segundos)
    y_original = df_filtrado['preco']

    # Criando novos pontos mais densos para suavizar a curva
    x_smooth = np.linspace(x_original.min(), x_original.max(), 300)  # 300 pontos suavizados
    spline = make_interp_spline(x_original, y_original, k=3)  # k=3 para suavização cúbica
    y_smooth = spline(x_smooth)

    # Plotando a linha suavizada
    ax.plot(pd.to_datetime(x_smooth, unit='s'), y_smooth, color=cor_primaria, linewidth=2, alpha=0.9)

    # Adicionando os pontos de máximo e mínimo
    ax.scatter(df_filtrado['data'].iloc[max_idx], max_val, color=cor_max, s=100, label="Máximo")
    ax.scatter(df_filtrado['data'].iloc[min_idx], min_val, color=cor_min, s=100, label="Mínimo")

    # Adicionando os rótulos com os valores ao lado dos pontos
    ax.text(df_filtrado['data'].iloc[max_idx], max_val, f"{max_val:.2f}", fontsize=12, verticalalignment='bottom', horizontalalignment='left', color=cor_max, fontweight='bold')
    ax.text(df_filtrado['data'].iloc[min_idx], min_val, f"{min_val:.2f}", fontsize=12, verticalalignment='top', horizontalalignment='left', color=cor_min, fontweight='bold')

    # Configuração do gráfico
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title('Série Temporal do Preço do Brent em US$', fontsize=18, fontweight="bold")
    ax.legend()

    # Exibir no Streamlit
    st.pyplot(fig)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 

    col1, col2, col3, col4, col5 = st.columns([1, 1.2, 1, 0.2, 1])

    # Botões principais (lado a lado)
    with col1:
        exibir_o_que_e_t = st.button('O que é uma série temporal?', key="btn_o_que_e_t")

    with col2:
        exibir_interpretar_t = st.button('Como interpretar uma série temporal?', key="btn_interpretar_t")

    with col3:
        exibir_porque_usar_t = st.button('Por que fazer uma série temporal do preço do Brent?', key="btn_porque_usar_t")

    if exibir_o_que_e_t:
            st.markdown('''
      Uma série temporal é uma sequência de observações registradas em intervalos de tempo regulares, como diário, semanal, mensal ou anual. Esses dados representam a evolução de uma variável ao longo do tempo.

        Exemplos de séries temporais incluem:

    Preço do petróleo Brent ao longo dos anos - Variações nos preços devido a crises econômicas, guerras e mudanças na oferta e demanda.

    Produção diária de barris de petróleo - Quantidade extraída por países ou empresas ao longo do tempo.

    Estoque global de petróleo - Níveis de armazenamento reportados mensalmente por organizações como a OPEP e a EIA.

    Consumo mundial de petróleo - Demanda de combustíveis fósseis por setor e região ao longo dos anos.

    Frete marítimo de petróleo bruto - Custos e volume transportado por rotas marítimas ao longo do tempo.
                    
            '''
            )
            st.button("x")

    if exibir_interpretar_t:
            st.markdown('''
        Ao analisar uma série temporal, é importante observar alguns componentes principais:

    1. Tendência (Trend)

    Refere-se ao movimento geral da série ao longo do tempo. Pode ser crescente, decrescente ou estável. Exemplo: 
    O preço do Brent teve uma tendência de alta entre 2000 e 2008.

    2. Sazonalidade (Seasonality)

    Padrões que se repetem em intervalos regulares. Exemplo: O consumo de energia elétrica pode ser maior no verão devido ao uso de ar-condicionado.

    3. Ciclos (Cycles)

    Flutuações de longo prazo que não têm uma periodicidade fixa. Exemplo: Crises econômicas que impactam o mercado a cada 5 a 10 anos.

    4. Ruído (Noise)

    São variações aleatórias que não seguem um padrão claro. Exemplo: Um pico abrupto no preço do petróleo devido a um evento inesperado.
              
            '''
            )
            st.button("x")

    if exibir_porque_usar_t:
            st.markdown('''
            As séries temporais são essenciais para diversas áreas porque ajudam a:

        1. Identificar padrões e tendências - Exemplo: Se o preço do petróleo sobe ao longo dos anos, pode indicar uma tendência de alta.
        
        2. Fazer previsões - Exemplo: Modelos podem prever o preço futuro do petróleo com base nos dados históricos.

        3. Detectar sazonalidades -  Embora exista sazonalidade na demanda por petróleo, ela não é rígida como a de outras comodities. Mas o clima, 
        atividade econômica e padrões de consumor energético podem dar aspectos sazonais na demanda pelo petróleo.
        
        4. Analisar o impacto de eventos externos - Exemplo: Pandemias, guerras e crises financeiras influenciam fortemente o preço do Brent.
        
        5. Apoiar decisões estratégicas - Apóia o planejamento de estoques, investimentos e estratégias de mercado.
            '''
            )
            st.button("x")

    st.markdown("<br><br>", unsafe_allow_html=True)  # Espaço 
    
#%% Conteúdo da Estacionariedade, Tendências e Sazonalidades
if pagina_selecionada == "Estacionariedade, Tendências e Sazonalidades":
        st.title("Análise de Estacionariedade, Tendências e Sazonalidades")
        st.markdown('''
        Nesta página, realizamos uma análise detalhada da série temporal do preço do Brent no período de **10/02/2015 a 10/02/2025**.
        Vamos explorar:
        - **Estacionariedade**: Verificar se a série é estacionária (média e variância constantes ao longo do tempo).
        - **Tendências**: Identificar padrões de crescimento ou declínio ao longo do tempo.
        - **Sazonalidades**: Detectar padrões repetitivos em intervalos regulares (ex.: mensal, anual).
        ''')

        # Filtrar dados para o período de 10/02/2015 a 10/02/2025
        data_inicio = pd.to_datetime("2015-02-10")
        data_fim = pd.to_datetime("2025-02-10")
        df_periodo = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)]
            
        st.subheader('Série Temporal do Preço do Brent (10/02/2015 - 10/02/2025)')
        fig_serie = px.line(df_periodo, x='data', y='preco', title='Preço do Brent ao Longo do Tempo',
                                    labels={'data': 'Data', 'preco': 'Preço (US$)'})
        fig_serie.update_layout(xaxis_title='Data', yaxis_title='Preço (US$)', hovermode='x unified')
        st.plotly_chart(fig_serie, use_container_width=True)
            
                # Análise de Estacionariedade (Teste de Dickey-Fuller Aumentado)
        st.subheader('Análise de Estacionariedade')
        st.markdown('''
                Para verificar se a série é estacionária, aplicamos o **Teste de Dickey-Fuller Aumentado (ADF)**.
                - **Hipótese Nula (H0)**: A série não é estacionária.
                - **Hipótese Alternativa (H1)**: A série é estacionária.
                Se o **p-valor** for menor que 0.05, rejeitamos H0 e consideramos a série estacionária.
                ''')
        
        # Aplicar o teste ADF
        resultado_adf = adfuller(df_periodo['preco'])
        p_valor = resultado_adf[1]
            
        st.write(f"**Estatística ADF:** {resultado_adf[0]:.4f}")
        st.write(f"**p-valor:** {p_valor:.4f}")
            
        if p_valor < 0.05:
            st.success("A série é **estacionária** (p-valor < 0.05).")
        else:
            st.warning("A série **não é estacionária** (p-valor ≥ 0.05).")
            
        # Análise de Tendências
        st.subheader('Análise de Tendências')
        st.markdown('''
                Para identificar tendências, aplicamos uma **decomposição da série temporal**.
                A decomposição separa a série em três componentes:
                - **Tendência**: Padrão de crescimento ou declínio ao longo do tempo.
                - **Sazonalidade**: Padrões repetitivos em intervalos regulares.
                - **Resíduo**: Variações aleatórias não explicadas pela tendência ou sazonalidade.
                ''')
            

                # Decomposição da série temporal
        decomposicao = seasonal_decompose(df_periodo.set_index('data')['preco'], model='additive', period=365)
            
                # Gráfico da Tendência
        st.write("**Tendência**")
        fig_tendencia = px.line(x=decomposicao.trend.index, y=decomposicao.trend, title='Tendência do Preço do Brent',
                                        labels={'x': 'Data', 'y': 'Preço (US$)'})
        fig_tendencia.update_layout(xaxis_title='Data', yaxis_title='Preço (US$)')
        st.plotly_chart(fig_tendencia, use_container_width=True)
            
                # Gráfico da Sazonalidade
        st.write("**Sazonalidade**")
        fig_sazonalidade = px.line(x=decomposicao.seasonal.index, y=decomposicao.seasonal, title='Sazonalidade do Preço do Brent',
                                        labels={'x': 'Data', 'y': 'Preço (US$)'})
        fig_sazonalidade.update_layout(xaxis_title='Data', yaxis_title='Preço (US$)')
        st.plotly_chart(fig_sazonalidade, use_container_width=True)
            
                # Gráfico dos Resíduos
        st.write("**Resíduos**")
        fig_residuos = px.line(x=decomposicao.resid.index, y=decomposicao.resid, title='Resíduos do Preço do Brent',
                                    labels={'x': 'Data', 'y': 'Preço (US$)'})
        fig_residuos.update_layout(xaxis_title='Data', yaxis_title='Preço (US$)')
        st.plotly_chart(fig_residuos, use_container_width=True)
            
                # Conclusão
        st.subheader('Conclusão')
        st.markdown(f'''
                - **Estacionariedade**: A série é estacionária? {"Sim" if p_valor < 0.05 else "Não"}.
                - **Tendência**: A tendência mostra um padrão de {"crescimento" if decomposicao.trend.mean() > 0 else "declínio"} ao longo do tempo.
                - **Sazonalidade**: Padrões sazonais são {"evidentes" if decomposicao.seasonal.std() > 0 else "fracos ou inexistentes"}.
                ''')
 
#%  Prophet
if pagina_selecionada == "Prophet":
    st.title("Modelagem com Prophet")
    st.markdown('''
    O **Prophet** é uma ferramenta desenvolvida pelo Facebook para previsão de séries temporais.
    Ele é especialmente útil para capturar:
    - **Tendências**: Crescimento ou declínio ao longo do tempo.
    - **Sazonalidades**: Padrões repetitivos (ex.: mensal, anual).
    - **Eventos Especiais**: Impacto de feriados ou crises.
    ''')

    # Filtrar dados para o período de 10/02/2015 a 10/02/2025
    data_inicio = pd.to_datetime("2015-02-10")
    data_fim = pd.to_datetime("2025-02-10")
    df_periodo = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)]

    if df_periodo.empty:
        st.warning("Nenhum dado disponível para o período selecionado.")
    else:
        # Preparar dados para o Prophet
        df_prophet = df_periodo[['data', 'preco']].rename(columns={'data': 'ds', 'preco': 'y'})

        # Dividir dados em treino e teste
        train_size = int(len(df_prophet) * 0.8)
        train, test = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]

        # Treinar o modelo Prophet
        st.write("**Treinando o modelo Prophet...aguarde, por favor.**")
        modelo_prophet = Prophet()
        modelo_prophet.fit(train)

        # Fazer previsões
        futuro = modelo_prophet.make_future_dataframe(periods=len(test))
        previsoes = modelo_prophet.predict(futuro)

        # Calcular métricas de desempenho
        rmse = np.sqrt(mean_squared_error(test['y'], previsoes['yhat'].iloc[train_size:]))
        mae = mean_absolute_error(test['y'], previsoes['yhat'].iloc[train_size:])

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")

        # Gráfico das Previsões
        st.write("**Previsões vs Valores Reais**")
        fig_previsoes = go.Figure()
        fig_previsoes.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Valores Reais'))
        fig_previsoes.add_trace(go.Scatter(x=test['ds'], y=previsoes['yhat'].iloc[train_size:], mode='lines', name='Previsões'))
        fig_previsoes.update_layout(title='Previsões do Preço do Brent com Prophet', xaxis_title='Data', yaxis_title='Preço (US$)')
        st.plotly_chart(fig_previsoes, use_container_width=True)

        # Componentes do Modelo
        st.write("**Componentes do Modelo Prophet**")
        fig_componentes = modelo_prophet.plot_components(previsoes)
        st.pyplot(fig_componentes)

        # Conclusão
        st.subheader('Conclusão')
        st.markdown(f'''
        - **RMSE**: {rmse:.2f} (Raiz do Erro Quadrático Médio): Mede a diferença média entre os valores reais e as previsões.
        - **MAE**: {mae:.2f} (Erro Absoluto Médio): Mede a diferença absoluta média entre os valores reais e as previsões.
        - **Tendência**: O modelo capturou a tendência de {"crescimento" if previsoes['trend'].mean() > 0 else "declínio"}.
        - **Sazonalidade**: Padrões sazonais foram {"evidentes" if previsoes['yearly'].std() > 0 else "fracos ou inexistentes"}.
        ''')

#%% LSTM
if pagina_selecionada == "LSTM":
    st.title("Modelagem com LSTM")
    st.markdown('''
    As **Redes Neurais Recorrentes (LSTM)** são poderosas para modelar séries temporais com padrões complexos e não lineares.
    Vamos implementar um modelo LSTM para prever o preço do Brent.
    ''')

    # Filtrar dados para o período de 10/02/2015 a 10/02/2025
    data_inicio = pd.to_datetime("2015-02-10")
    data_fim = pd.to_datetime("2025-02-10")
    df_periodo = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)]

    if df_periodo.empty:
        st.warning("Nenhum dado disponível para o período selecionado.")
    else:
        # Preparar dados para o LSTM
        dados = df_periodo['preco'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dados_escalados = scaler.fit_transform(dados)

        # Criar sequências temporais
        def criar_sequencias(dados, janela_temporal):
            X, y = [], []
            for i in range(janela_temporal, len(dados)):
                X.append(dados[i-janela_temporal:i, 0])
                y.append(dados[i, 0])
            return np.array(X), np.array(y)

        janela_temporal = 60
        X, y = criar_sequencias(dados_escalados, janela_temporal)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Dividir dados em treino e teste
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Construir o modelo LSTM
        st.write("**Construindo e treinando o modelo LSTM...aguarde, por favor**")
        modelo_lstm = Sequential()
        modelo_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        modelo_lstm.add(LSTM(units=50, return_sequences=False))
        modelo_lstm.add(Dense(units=25))
        modelo_lstm.add(Dense(units=1))

        modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')
        modelo_lstm.fit(X_train, y_train, batch_size=32, epochs=20)

        # Fazer previsões
        previsoes = modelo_lstm.predict(X_test)
        previsoes = scaler.inverse_transform(previsoes)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calcular métricas de desempenho
        rmse = np.sqrt(mean_squared_error(y_test, previsoes))
        mae = mean_absolute_error(y_test, previsoes)

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")

        # Gráfico das Previsões
        st.write("**Previsões vs Valores Reais**")
        fig_previsoes = go.Figure()
        fig_previsoes.add_trace(go.Scatter(x=df_periodo['data'].iloc[train_size+janela_temporal:], y=y_test.flatten(), mode='lines', name='Valores Reais'))
        fig_previsoes.add_trace(go.Scatter(x=df_periodo['data'].iloc[train_size+janela_temporal:], y=previsoes.flatten(), mode='lines', name='Previsões'))
        fig_previsoes.update_layout(title='Previsões do Preço do Brent com LSTM', xaxis_title='Data', yaxis_title='Preço (US$)')
        st.plotly_chart(fig_previsoes, use_container_width=True)

        # Conclusão
        st.subheader('Conclusão')
        st.markdown(f'''
        - **RMSE**: {rmse:.2f} (Raiz do Erro Quadrático Médio): Mede a diferença média entre os valores reais e as previsões.
        - **MAE**: {mae:.2f} (Erro Absoluto Médio): Mede a diferença absoluta média entre os valores reais e as previsões.
        - **Desempenho**: O modelo LSTM capturou padrões complexos na série temporal.
        ''')

if pagina_selecionada == "Sobre o Desafio":
    st.title("Sobre o Desafio")
    st.markdown('''
    O desafio é o Tech Challenge da fase 4, Data Viz and Production Models, da Pos Tech em Data Analytics da Fiap+Alura, da turma 6DTA. 
    
    Os alunos devem construir uma aplicação web para análise e previsão do preço do barril de petróleo Brent, utilizando técnicas de análise de séries temporais e modelagem preditiva. 
                
    O projeto envolve deploy de um MVP (Minimum Viable Product), ou Produto Mínimo Viável, que é uma versão simplificada de um produto ou serviço.
    O MPV  contém apenas as funcionalidades essenciais para ser testado no mercado com o menor esforço e custo possíveis. 
    
    O objetivo do MVP é validar uma ideia, coletar feedback dos usuários reais e fazer ajustes antes de um lançamento completo.
                
    ''')
if pagina_selecionada == "Sobre o Desenvolvedor":
    st.title("Sobre o Desenvolvedor")
    st.markdown('''
    O **grupo 49** desenvolveu esta aplicação para a fase 4 da turma 6DTA, do curso de Pós-Graduação em Data Analytics da FIAP/Alura. O grupo contou com um integrante dedicado:

    - **Rogério Abramo Alves Pretti**, RM 357672.

    Para saber mais sobre o curso, acesse o link abaixo:

    Pós Tech em Data Analytics da FIAP+Alura
    ''')
#%% Video Explicativo
if pagina_selecionada == "Video Explicativo":
    st.title("Video Descritivo")
    
    # Link do vídeo do YouTube
    video_url = "https://youtu.be/a12t0Uigz3E"
    
    # Exibe o vídeo
    st.video(video_url)
