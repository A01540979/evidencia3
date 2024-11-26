import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuración general de la página
st.set_page_config(page_title="Ternium - Cuellos de Botella", page_icon="CamporaFavicon.ico", layout="wide")

# CSS personalizado para mejorar el diseño
st.markdown("""
    <style>
        /* Sidebar en el lado derecho */
        .css-1y4p8pa {
            order: 2;  /* Mueve el sidebar al lado derecho */
        }
        .css-e76fdz {
            order: 1;  /* Mueve el contenido principal a la izquierda */
        }
        /* Fondo del contenido principal */
        .main {
            background-color: #f0f0f5; /* Gris claro */
        }
        /* Barra de encabezado */
        .header-bar {
            background-color: #f25c24; /* Naranja */
            color: white;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            padding: 10px 0;
            border-radius: 8px;
        }
        /* Títulos y subtítulos */
        h1, h2, h3, h4 {
            color: #333333; /* Texto oscuro */
            font-family: Arial, sans-serif;
        }
        h2 {
            margin-top: 20px;
            margin-bottom: 10px;
        }
        h3 {
            margin-bottom: 5px;
        }
        /* Tablas */
        .dataframe {
            border: 1px solid #cccccc; /* Borde claro */
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Encabezado
col_header, col_image = st.columns([4, 1], gap="medium")
with col_header:
    st.markdown('<div class="header-bar">Ternium - Dashboard de Cuellos de Botella</div>', unsafe_allow_html=True)
with col_image:
    st.image("Evidencia presentación 1.png", width=88)  # Logo reducido

# Cargar CSV
# Enlace directo del archivo en Google Drive
url = "https://drive.google.com/uc?id=1ooiYlR1tpfrdT2kkeLYsKbQ_NyYOtIJ5"

try:
    df = pd.read_csv(url)
    st.success("Datos cargados exitosamente desde Google Drive")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")

# Convertir columna a datetime
if 'Taper fecha_date' in df.columns:
    df['Taper fecha_date'] = pd.to_datetime(df['Taper fecha_date'], format='%d/%m/%Y')
else:
    st.error("La columna 'Taper fecha_date' no está en los datos.")

# Mapear categorías
if 'Dureza_categoria' in df.columns:
    df['Dureza_categoria'] = df['Dureza_categoria'].map({0: "Soft", 1: "Medium", 2: "Hard"})
else:
    st.warning("La columna 'Dureza_categoria' no está en los datos.")

# Sidebar para filtros y navegación
with st.sidebar:
    st.image("Ternium_Logo.svg.png", width=200)  # Ajustado el uso de width
    st.header('Navegación')
    st.markdown('[Dashboard de Análisis](#dashboard-de-análisis)', unsafe_allow_html=True)
    st.markdown('[Datos Filtrados](#datos-filtrados)', unsafe_allow_html=True)

    st.header('Filtros')

    # Botón para limpiar filtros
    if st.button("Limpiar Filtros"):
        st.session_state["demorado_filter"] = 0
        st.session_state["interrupcion_filter"] = 0
        st.session_state["dureza_seleccionada"] = ["Todos"]
        st.session_state["meses_seleccionados"] = ["Todos los meses"]
        st.session_state["bottleneck_seleccionados"] = ["Todos"]

    # Filtro para demorado
    demorado_filter = st.selectbox(
        '¿Demorado?',
        [0, 1],
        index=st.session_state.get("demorado_filter", 0),  # Usa el estado guardado o valor inicial
        format_func=lambda x: 'Sí (1)' if x == 1 else 'No (0)'
    )
    st.session_state["demorado_filter"] = demorado_filter  # Actualiza el estado

    # Filtro para interrupción
    interrupcion_filter = st.selectbox(
        '¿Interrupción?',
        [0, 1],
        index=st.session_state.get("interrupcion_filter", 0),
        format_func=lambda x: 'Sí (1)' if x == 1 else 'No (0)'
    )
    st.session_state["interrupcion_filter"] = interrupcion_filter

    # Filtro para dureza
    st.subheader("Filtrar por Dureza")
    if 'Dureza_categoria' in df.columns:
        dureza_seleccionada = st.multiselect(
            "Selecciona las categorías:",
            ["Todos"] + list(df['Dureza_categoria'].unique()),
            default=st.session_state.get("dureza_seleccionada", ["Todos"])
        )
        st.session_state["dureza_seleccionada"] = dureza_seleccionada

    # Filtro para meses
    st.subheader("Filtrar por Meses")
    meses_seleccionados = st.multiselect(
        "Selecciona los meses:",
        ["Todos los meses", "Abril", "Mayo", "Junio", "Julio", "Agosto"],
        default=st.session_state.get("meses_seleccionados", ["Todos los meses"])
    )
    st.session_state["meses_seleccionados"] = meses_seleccionados

    # Filtro para Bottleneck
    st.subheader("Filtrar por Bottleneck")
    if 'Bottleneck' in df.columns:
        bottleneck_seleccionados = st.multiselect(
            "Selecciona las categorías:",
            ["Todos"] + list(df['Bottleneck'].unique()),
            default=st.session_state.get("bottleneck_seleccionados", ["Todos"])
        )
        st.session_state["bottleneck_seleccionados"] = bottleneck_seleccionados
    else:
        st.warning("La columna 'Bottleneck' no está en los datos.")

# Filtrar datos
filtered_df = df[(df['Demorado'] == demorado_filter) & (df['Interrupcion'] == interrupcion_filter)]
if "Todos" not in dureza_seleccionada:
    filtered_df = filtered_df[filtered_df['Dureza_categoria'].isin(dureza_seleccionada)]
if "Todos los meses" not in meses_seleccionados:
    meses_map = {"Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8}
    meses_numeros = [meses_map[mes] for mes in meses_seleccionados]
    filtered_df = filtered_df[filtered_df['Taper fecha_date'].dt.month.isin(meses_numeros)]
if "Todos" not in bottleneck_seleccionados:
    filtered_df = filtered_df[filtered_df['Bottleneck'].isin(bottleneck_seleccionados)]

# Sección de Dashboard de Análisis
st.markdown('<div id="dashboard-de-análisis"></div>', unsafe_allow_html=True)
st.title("Dashboard de Análisis")

# Función para graficar frecuencia de cuellos de botella con contornos negros en las barras
def plot_bottleneck_frequency(filtered_df, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=filtered_df,
        x='Bottleneck',
        palette='YlOrBr',
        edgecolor='black',  # Contorno negro en las barras
        ax=ax
    )
    ax.set_title(title, fontsize=14, color="#FFD700")
    ax.set_xlabel('Bottleneck', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    sns.despine(left=True, bottom=True)
    return fig
# Gráficas
cols = st.columns(3)

with cols[0]:
    st.subheader("Frecuencia de Cuellos de Botella")
    if not filtered_df.empty:
        st.pyplot(plot_bottleneck_frequency(filtered_df, "Frecuencia por Bottleneck"))
    else:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")

for i, col in enumerate(cols[1:], 1):
    with col:
        st.subheader(f"Gráfico Falso {i}")
        st.pyplot(plot_bottleneck_frequency(filtered_df, f"Gráfico {i}"))

# Sección de Datos Filtrados
st.markdown('<div id="datos-filtrados"></div>', unsafe_allow_html=True)
st.title("Datos Filtrados")
st.dataframe(filtered_df)
# Agregar gráfica de línea temporal para "Delay Time" vs "Taper fecha_date"
# Agregar gráfica de línea temporal para "Delay Time" vs "Taper fecha_date" diferenciando por "Bottleneck"
st.subheader("Tendencia de Delay Time en el Tiempo por Bottleneck")
if not filtered_df.empty:
    # Verificar que las columnas necesarias existan
    if 'Delay Time' in filtered_df.columns and 'Taper fecha_date' in filtered_df.columns and 'Bottleneck' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        filtered_df = filtered_df.sort_values('Taper fecha_date')  # Ordenar por fecha

        # Diferenciar por Bottleneck si no se seleccionó "Todos"
        if "Todos" not in bottleneck_seleccionados:
            for bottleneck in filtered_df['Bottleneck'].unique():
                subset = filtered_df[filtered_df['Bottleneck'] == bottleneck]
                ax.plot(
                    subset['Taper fecha_date'], 
                    subset['Delay Time'], 
                    marker='o', 
                    label=bottleneck, 
                    linestyle='-', 
                    linewidth=1.5
                )
        else:
            # Graficar todo en un solo color si "Todos" está seleccionado
            ax.plot(
                filtered_df['Taper fecha_date'], 
                filtered_df['Delay Time'], 
                marker='o', 
                color='orange', 
                label="Todos los Bottleneck", 
                linestyle='-', 
                linewidth=1.5
            )
        
        ax.set_title("Tendencia de Delay Time en el Tiempo", fontsize=14, color="#333333")
        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Delay Time", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Bottleneck", loc="upper left", fontsize=10)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    else:
        st.warning("No se encuentran las columnas necesarias ('Delay Time', 'Taper fecha_date', y/o 'Bottleneck') en los datos filtrados.")
else:
    st.warning("No hay datos que coincidan con los filtros seleccionados para graficar la tendencia de 'Delay Time'.")