# import streamlit as st
# import folium
# from streamlit_folium import folium_static
# import pandas as pd
# from pyspark.sql import SparkSession
# import plotly.express as px
# from geopy.geocoders import Nominatim


# # Configurar sessão Spark
# spark = SparkSession.builder.getOrCreate()

# # Carregar dados processados
# @st.cache_data
# def load_data():
#     df = spark.read.parquet("dados_processados.parquet").toPandas()
#     return df

# df = load_data()

# # Título do Dashboard
# st.title("Análise Educacional por Município - ENEM 2022")

# geolocator = Nominatim(user_agent="dashboard_educacao")

# def get_coordinates(municipio, uf):
#     location = geolocator.geocode(f"{municipio}, {uf}, Brasil")
#     return (location.latitude, location.longitude) if location else (None, None)

# # Filtros na barra lateral
# with st.sidebar:
#     st.header("Filtros")
    
#     # Filtro por UF
#     ufs = df["SG_UF"].unique().tolist()
#     selected_uf = st.multiselect("Selecione o Estado:", options=ufs, default=ufs[:3])
    
#     # Filtro por nota mínima
#     min_nota = float(df["nota_media"].min())
#     max_nota = float(df["nota_media"].max())
#     nota_range = st.slider("Intervalo da Nota Média:", min_nota, max_nota, (min_nota, max_nota))
    
#     # Filtro por infraestrutura
#     with st.expander("Infraestrutura Escolar"):
#         show_lab = st.checkbox("Com Laboratório de Informática", value=True)
#         show_biblioteca = st.checkbox("Com Biblioteca", value=True)

# # Aplicar filtros
# filtered_df = df[
#     (df["SG_UF"].isin(selected_uf)) &
#     (df["nota_media"] >= nota_range[0]) &
#     (df["nota_media"] <= nota_range[1])
# ]

# if show_lab:
#     filtered_df = filtered_df[filtered_df["IN_LABORATORIO_INFORMATICA"] == 1]
# if show_biblioteca:
#     filtered_df = filtered_df[filtered_df["IN_BIBLIOTECA"] == 1]

# # Mapa
# st.header("Mapa de Notas Médias por Município")
# m = folium.Map(location=[-15.788497, -47.879873], zoom_start=4)

# # Adicionar marcadores
# for idx, row in filtered_df.iterrows():
#     folium.CircleMarker(
#         location=[row["LATITUDE"]],  # Substituir por coluna real
#         radius=5,
#         color="#3186cc",
#         fill=True,
#         fill_color="#3186cc",
#         popup=f"""
#         Município: {row["NO_MUNICIPIO"]}<br>
#         Nota Média: {row["nota_media"]:.1f}<br>
#         Proporção Feminina: {row["proporcao_feminino"]:.2%}
#         """
#     ).add_to(m)

# folium_static(m)

# # Gráfico de Dispersão
# st.header("Relação entre Variáveis")
# x_axis = st.selectbox("Eixo X:", options=df.columns, index=df.columns.get_loc("proporcao_feminino"))
# y_axis = st.selectbox("Eixo Y:", options=df.columns, index=df.columns.get_loc("nota_media"))

# fig = px.scatter(
#     filtered_df,
#     x=x_axis,
#     y=y_axis,
#     color="SG_UF",
#     hover_name="NO_MUNICIPIO",
#     trendline="ols"
# )
# st.plotly_chart(fig)

# # Matriz de Correlação
# st.header("Correlações")
# corr_vars = st.multiselect("Selecione variáveis:", options=df.columns, default=["nota_media", "proporcao_feminino", "V01327"])
# corr_matrix = filtered_df[corr_vars].corr()

# fig_corr = px.imshow(
#     corr_matrix,
#     labels=dict(x="Variáveis", y="Variáveis", color="Correlação"),
#     x=corr_matrix.columns,
#     y=corr_matrix.columns,
#     color_continuous_scale="RdBu",
#     range_color=[-1, 1]
# )
# st.plotly_chart(fig_corr)

# # Métricas
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Municípios Filtrados", len(filtered_df))
# with col2:
#     avg_score = filtered_df["nota_media"].mean()
#     st.metric("Nota Média", f"{avg_score:.1f}")
# with col3:
#     fem_ratio = filtered_df["proporcao_feminino"].mean()
#     st.metric("Proporção Feminina Média", f"{fem_ratio:.2%}")


import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(page_title="Dashboard Educação", layout="wide")
st.title("📊 Análise Educacional por Município - ENEM 2022")

# Configurar sessão Spark
spark = SparkSession.builder.getOrCreate()

# Função para carregar e processar dados
@st.cache_data
def load_data():
    try:
        df = spark.read.parquet("dados_processados.parquet").toPandas()
        
        # Verificar se as coordenadas já existem
        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            raise Exception("Coordenadas não encontradas - Iniciando geocoding")
            
    except Exception as e:
        st.warning(f"{str(e)}")
        
        # Carregar dados brutos se o parquet não existir
        df = spark.read.parquet("dados_brutos.parquet").toPandas()
        
        # Configurar geolocator
        geolocator = Nominatim(user_agent="dashboard_educacao_v1")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        # Adicionar colunas de coordenadas
        df['LATITUDE'] = None
        df['LONGITUDE'] = None
        
        # Barra de progresso
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_municipios = len(df)
        
        for index, row in df.iterrows():
            try:
                location = geocode(f"{row['NO_MUNICIPIO']}, {row['SG_UF']}, Brasil")
                if location:
                    df.at[index, 'LATITUDE'] = location.latitude
                    df.at[index, 'LONGITUDE'] = location.longitude
            except Exception as geocode_error:
                st.error(f"Erro no município {row['NO_MUNICIPIO']}: {str(geocode_error)}")
            
            # Atualizar progresso
            progress = (index + 1) / total_municipios
            progress_bar.progress(progress)
            progress_text.text(f"Processando municípios: {index+1}/{total_municipios} ({progress:.1%})")
        
        # Remover entradas sem coordenadas
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        
        # Salvar dados processados
        spark.createDataFrame(df).write.parquet("dados_processados.parquet", mode="overwrite")
        
        progress_bar.empty()
        progress_text.empty()
    
    return df

# Carregar dados
df = load_data()

# Configurar página

# Sidebar com filtros
with st.sidebar:
    st.header("⚙️ Filtros")
    
    # Filtro por UF
    ufs = sorted(df["SG_UF"].unique().tolist())
    selected_uf = st.multiselect("Selecione o Estado:", options=ufs, default=ufs[:2])
    
    # Filtro por nota
    min_nota = float(df["nota_media"].min())
    max_nota = float(df["nota_media"].max())
    nota_range = st.slider("Intervalo da Nota Média:", min_nota, max_nota, (min_nota, max_nota))
    
    # Filtro por infraestrutura
    with st.expander("🏫 Infraestrutura Escolar"):
        show_lab = st.checkbox("Com Laboratório de Informática", value=True)
        show_biblioteca = st.checkbox("Com Biblioteca", value=True)

# Aplicar filtros
filtered_df = df[
    (df["SG_UF"].isin(selected_uf)) &
    (df["nota_media"] >= nota_range[0]) &
    (df["nota_media"] <= nota_range[1])
]

if show_lab:
    filtered_df = filtered_df[filtered_df["IN_LABORATORIO_INFORMATICA"] == 1]
if show_biblioteca:
    filtered_df = filtered_df[filtered_df["IN_BIBLIOTECA"] == 1]

# Seção de métricas
st.header("📈 Métricas Gerais")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Municípios Filtrados", len(filtered_df))
with col2:
    avg_score = filtered_df["nota_media"].mean()
    st.metric("Nota Média", f"{avg_score:.1f}")
with col3:
    fem_ratio = filtered_df["proporcao_feminino"].mean()
    st.metric("👩 Proporção Feminina Média", f"{fem_ratio:.2%}")

# Seção do mapa
st.header("🗺️ Mapa de Desempenho")
m = folium.Map(location=[-15.788497, -47.879873], zoom_start=4)

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=5,
        color='#3186cc',
        fill=True,
        fill_color='#3186cc',
        popup=folium.Popup(
            f"""<b>{row['NO_MUNICIPIO']}</b><br>
            UF: {row['SG_UF']}<br>
            Nota Média: {row['nota_media']:.1f}<br>
            Proporção Feminina: {row['proporcao_feminino']:.2%}<br>
            População Branca: {row['V01317']}<br>
            Laboratório: {'✅' if row['IN_LABORATORIO_INFORMATICA'] else '❌'}
            """,
            max_width=300
        )
    ).add_to(m)

folium_static(m, width=1200, height=600)

# Seção de gráficos
st.header("📊 Análise de Relações")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Relação entre Variáveis")
    x_axis = st.selectbox("Eixo X:", options=df.columns, index=df.columns.get_loc("proporcao_feminino"))
    y_axis = st.selectbox("Eixo Y:", options=df.columns, index=df.columns.get_loc("nota_media"))
    
    fig_scatter = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color="SG_UF",
        hover_name="NO_MUNICIPIO",
        trendline="ols",
        labels={
            x_axis: x_axis.replace("_", " ").title(),
            y_axis: y_axis.replace("_", " ").title()
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("Matriz de Correlação")
    corr_vars = st.multiselect(
        "Selecione variáveis para correlação:",
        options=df.columns,
        default=["nota_media", "proporcao_feminino", "V01327", "IN_LABORATORIO_INFORMATICA"]
    )
    
    if corr_vars:
        corr_matrix = filtered_df[corr_vars].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(x="Variáveis", y="Variáveis", color="Correlação"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            text_auto=".2f"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Selecione pelo menos duas variáveis para ver as correlações.")

# Notas importantes
st.markdown("""
---
**Notas:**
1. As coordenadas são aproximadas usando o serviço Nominatim
2. Municípios sem dados de localização foram excluídos da análise
3. Dados atualizados conforme o Censo 2022
""")