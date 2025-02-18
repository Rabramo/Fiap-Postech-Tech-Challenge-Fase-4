# Importação das bibliotecas necessárias
import streamlit as st  # Biblioteca principal para criar aplicativos interativos
import pandas as pd  # Manipulação de dados em tabelas DataFrame
import locale  # Configuração de localidade (moeda, número, data, etc.)
import matplotlib.pyplot as plt  # Biblioteca para visualização de gráficos
import seaborn as sns  # Biblioteca para visualização de dados avançada
from datetime import datetime, timedelta  # Manipulação de datas e períodos de tempo
import numpy as np  # Cálculos matemáticos avançados e manipulação de arrays
from scipy.interpolate import make_interp_spline  # Interpolação para suavizar gráficos

# Configuração do Streamlit
# Definir a largura da página do app para 'auto' que Dinamicamente ajusta entre "centered" e "wide", dependendo do tamanho da tela do usuário.
st.set_page_config(layout="wide")
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


# Criando DF a partir do parquet com dados do Brent do Ipea (https://www.ipeadata.gov.br/Default.aspx), usando r, raw string para evitar problemas com barras
df = pd.read_parquet(r'C:\Users\17609633801\Downloads\git\Fiap-Postech-Tech-Challenge-Fase-4\ipea_brent_20250217.parquet')

# Título da páginastrea
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

# exibir estatísticas básicas
st.title("Previsão do Preço do Brent")
st.write("Analisando a evolução do preço do petróleo Brent e aplicando modelos de previsão.")

# Obtendo os anos únicos do DataFrame filtrado
anos_filtrados = sorted(df_filtrado["ano"].unique())

# Verificar se há anos filtrados e formatar a exibição
if len(anos_filtrados) > 1:
    anos_exibidos = f"{anos_filtrados[0]} - {anos_filtrados[-1]}"
else:
    anos_exibidos = f"{anos_filtrados[0]}" if anos_filtrados else "Nenhum ano disponível"

# Gerar estatísticas descritivas
desc = df_filtrado["preco"].describe()

# Renomear os índices para português
desc_traduzido = desc.rename(index={
    "count": "Contagem",
    "mean": "Média",
    "std": "Desvio Padrão",
    "min": "Mínimo",
    "25%": "1º Quartil (25%)",
    "50%": "Mediana (50%)",
    "75%": "3º Quartil (75%)",
    "max": "Máximo"
}).to_frame()

# 🔹 Converter "Contagem" para inteiro SEM casas decimais
desc_traduzido.loc["Contagem"] = int(desc_traduzido.loc["Contagem"])

# 🔹 Converter os demais valores para float com apenas 2 casas decimais e formatar no padrão brasileiro
for indice in desc_traduzido.index:
    if indice != "Contagem":  # Garantir que apenas floats são arredondados
        valor = round(float(desc_traduzido.loc[indice]), 2)
        desc_traduzido.loc[indice] = locale.format_string("%.2f", valor, grouping=True)

# 🔹 Remover o nome da coluna ("preco")
desc_traduzido.columns = [""]

# 🔹 Obter o período selecionado corretamente
anos_filtrados = sorted(df_filtrado["ano"].unique())
periodo = f"{anos_filtrados[0]} - {anos_filtrados[-1]}" if len(anos_filtrados) > 1 else f"{anos_filtrados[0]}"

# 🔹 Exibir título corretamente com o período selecionado
st.subheader(f"Estatísticas do Preço do Brent ({periodo})")

# 🔹 Aplicar CSS para remover o cabeçalho da tabela e centralizar a exibição
st.markdown(
    """
    <style>
        table {
            width: 40% !important;
            margin: auto;
        }
        thead { display: none; } /* Remove cabeçalho */
        tbody tr td {
            text-align: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Exibir a tabela corretamente formatada
st.table(desc_traduzido)

