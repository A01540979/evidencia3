import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="Ternium - Cuellos de Botella", page_icon="CamporaFavicon.ico", layout="wide")

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
    st.image("Evidencia presentación 1.png", width=88)

# Cargar CSV
df = pd.read_csv('BDD_Final.csv')

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
    st.image("Ternium_Logo.svg.png", width=200) 
    st.header('Navegación')
    st.markdown('[Dashboard de Análisis](#dashboard-de-análisis)', unsafe_allow_html=True)
    st.markdown('[Línea de tiempo](#linea-de-tiempo)', unsafe_allow_html=True)
    st.markdown('[Datos Filtrados](#datos-filtrados)', unsafe_allow_html=True)

    st.header('Filtros')

    # Inicializar valores en st.session_state si no existen
    if "demorado_filter" not in st.session_state:
        st.session_state["demorado_filter"] = "Todas"
    if "interrupcion_filter" not in st.session_state:
        st.session_state["interrupcion_filter"] = "Todas"
    if "dureza_seleccionada" not in st.session_state:
        st.session_state["dureza_seleccionada"] = ["Todos"]
    if "meses_seleccionados" not in st.session_state:
        st.session_state["meses_seleccionados"] = ["Todos los meses"]
    if "bottleneck_seleccionados" not in st.session_state:
        st.session_state["bottleneck_seleccionados"] = ["Todos"]

    # Botón para limpiar filtros
    if st.button("Limpiar Filtros"):
        st.session_state["demorado_filter"] = "Todas"
        st.session_state["interrupcion_filter"] = "Todas"
        st.session_state["dureza_seleccionada"] = ["Todos"]
        st.session_state["meses_seleccionados"] = ["Todos los meses"]
        st.session_state["bottleneck_seleccionados"] = ["Todos"]

    # Filtro para demorado
    demorado_filter = st.selectbox(
        '¿Demorado?',
        ['Todas', 0, 1],
        index=['Todas', 0, 1].index(st.session_state["demorado_filter"]),
        format_func=lambda x: 'Todas' if x == 'Todas' else ('Sí (1)' if x == 1 else 'No (0)'),
        key="demorado_filter"
    )

    # Filtro para interrupción
    interrupcion_filter = st.selectbox(
        '¿Interrupción?',
        ['Todas', 0, 1],
        index=['Todas', 0, 1].index(st.session_state["interrupcion_filter"]),
        format_func=lambda x: 'Todas' if x == 'Todas' else ('Sí (1)' if x == 1 else 'No (0)'),
        key="interrupcion_filter"
    )

    # Filtro para dureza
    st.subheader("Filtrar por Dureza")
    if 'Dureza_categoria' in df.columns:
        dureza_seleccionada = st.multiselect(
            "Selecciona las categorías:",
            ["Todos"] + list(df['Dureza_categoria'].unique()),
            default=st.session_state.get("dureza_seleccionada", ["Todos"]),
            key="dureza_seleccionada"
        )

    # Filtro para meses
    st.subheader("Filtrar por Meses")
    meses_seleccionados = st.multiselect(
        "Selecciona los meses:",
        ["Todos los meses", "Abril", "Mayo", "Junio", "Julio", "Agosto"],
        default=st.session_state.get("meses_seleccionados", ["Todos los meses"]),
        key="meses_seleccionados"
    )

    # Filtro para Bottleneck
    st.subheader("Filtrar por Cuello de Botella")
    if 'Bottleneck' in df.columns:
        bottleneck_seleccionados = st.multiselect(
            "Selecciona las categorías:",
            ["Todos"] + list(df['Bottleneck'].unique()),
            default=st.session_state.get("bottleneck_seleccionados", ["Todos"]),
            key="bottleneck_seleccionados"
        )
    else:
        st.warning("La columna 'Bottleneck' no está en los datos.")
# Filtrar datos
filtered_df = df.copy()

# Filtro para Demorado
if demorado_filter != 'Todas':
    filtered_df = filtered_df[filtered_df['Demorado'] == demorado_filter]

# Filtro para Interrupción
if interrupcion_filter != 'Todas':
    filtered_df = filtered_df[filtered_df['Interrupcion'] == interrupcion_filter]

# Filtro para Dureza
if "Todos" not in dureza_seleccionada:
    filtered_df = filtered_df[filtered_df['Dureza_categoria'].isin(dureza_seleccionada)]

# Filtro para Meses
if "Todos los meses" not in meses_seleccionados:
    meses_map = {"Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8}
    meses_numeros = [meses_map[mes] for mes in meses_seleccionados]
    filtered_df = filtered_df[filtered_df['Taper fecha_date'].dt.month.isin(meses_numeros)]

# Filtro para Bottleneck
if "Todos" not in bottleneck_seleccionados:
    filtered_df = filtered_df[filtered_df['Bottleneck'].isin(bottleneck_seleccionados)]


# Función cuellos de botella 
def plot_bottleneck_frequency(filtered_df, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    barplot = sns.countplot(
        data=filtered_df,
        x='Bottleneck',
        palette='YlOrBr',
        edgecolor='black',
        ax=ax
    )
    
    for container in barplot.containers:
        ax.bar_label(container, fmt=lambda x: f'{x/1000:.1f}K', label_type='edge', fontsize=10, color='black', padding=3)

    # Configurarción etiquetas
    ax.set_xlabel('Cuellos de Botella', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    
    # Configurar el estilo
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    sns.despine(left=True, bottom=True)
    
    return fig

# Función Delay Concept--------------------
def plot_delay_concept_bubble(filtered_df, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    top_12_delays = filtered_df['Delay Concept'].value_counts().head(12)
    top_12_df = pd.DataFrame({
        'Delay Concept': top_12_delays.index,
        'Frecuencia': top_12_delays.values
    })

    x_coords = np.full(len(top_12_df), 0.5) 

    y_coords = np.linspace(len(top_12_df), 0, len(top_12_df))

    max_size = 1500  
    min_size = 150
    bubble_sizes = (top_12_df['Frecuencia'] - top_12_df['Frecuencia'].min()) / \
                   (top_12_df['Frecuencia'].max() - top_12_df['Frecuencia'].min()) * \
                   (max_size - min_size) + min_size

    scatter = ax.scatter(
        x=x_coords,
        y=y_coords,
        s=bubble_sizes,
        alpha=0.7,
        c=sns.color_palette("YlOrBr", len(top_12_df)),
        edgecolor='black'
    )

    for i, (label, freq) in enumerate(zip(top_12_df['Delay Concept'], top_12_df['Frecuencia'])):
        ax.text(
            x_coords[i], 
            y_coords[i] + 0.2,  
            label, 
            ha='center', 
            va='center', 
            fontsize=7, 
            color="black", 
            fontweight='bold'
        )
        ax.text(
            x_coords[i], 
            y_coords[i] - 0.2, 
            f"{freq}K", 
            ha='center', 
            va='center', 
            fontsize=6, 
            color="black"
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(top_12_df) + 1)  
    ax.grid(False)
    sns.despine(left=True, bottom=True)

    return fig

def plot_feature_importance(df, Demorado, title):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df_encoded = df.select_dtypes(exclude=['datetime64[ns]']).copy()
    
    X = df_encoded.drop(columns=["Demorado", "Delay Time", "Interrupcion"])
    y = df_encoded["Demorado"]
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.select_dtypes(include=['number']).fillna(0)
    y = y.fillna(0)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=importances, 
        x="Importance", 
        y="Feature", 
        palette="YlOrBr", 
        edgecolor="black", 
        ax=ax
    )
    ax.set_xlabel("Nivel de Importancia (%)", fontsize=12)
    ax.set_ylabel("Características", fontsize=12)
    ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])
    sns.despine(left=True, bottom=True)
    return fig

# Gráficas
cols = st.columns(3)

with cols[0]:
    st.subheader("Frecuencia: Cuellos de Botella")
    if not filtered_df.empty:
        st.pyplot(plot_bottleneck_frequency(filtered_df, "Frecuencia por Cuello de Bottella"))
    else:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")

with cols[1]:
    st.subheader("Frecuencia: 'Delay Concept'")
    if not filtered_df.empty and 'Delay Concept' in filtered_df.columns:
        st.pyplot(plot_delay_concept_bubble(filtered_df, "Frecuencia de Delay Concept (Top 15)"))
    else:
        st.warning("No hay datos que coincidan con los filtros seleccionados o 'Delay Concept' no está en los datos.")


with cols[2]:  
    st.subheader("Top 10 Variables Significativas (Random Forest)")
    if not filtered_df.empty and "Interrupcion" in filtered_df.columns:
        st.pyplot(plot_feature_importance(filtered_df, "Interrupcion", "Top 10 Variables más Significativas"))
    else:
        st.warning("No hay datos suficientes o la columna 'Interrupcion' no está disponible.")

#Linea temporal
st.markdown('<div id="linea-de-tiempo"></div>', unsafe_allow_html=True)
st.subheader("Tendencias en el Tiempo")
if not filtered_df.empty:

    if 'Delay Time' in filtered_df.columns and 'Taper fecha_date' in filtered_df.columns and 'Bottleneck' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        filtered_df = filtered_df.sort_values('Taper fecha_date')  
        
        if "Todos" not in bottleneck_seleccionados:
            for bottleneck in filtered_df['Bottleneck'].unique():
                subset = filtered_df[filtered_df['Bottleneck'] == bottleneck]
                ax.plot(
                    subset['Taper fecha_date'], 
                    subset['Delay Time'] / 3600, 
                    marker='o', 
                    label=bottleneck, 
                    linestyle='-', 
                    linewidth=1.5
                )
        else:
            ax.plot(
                filtered_df['Taper fecha_date'], 
                filtered_df['Delay Time'] / 3600, 
                marker='o', 
                color='orange', 
                label="Todos los Cuellos de Bottella", 
                linestyle='-', 
                linewidth=1.5
            )
        
        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Delay Time (horas)", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Cuellos de botella", loc="upper left", fontsize=10)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    else:
        st.warning("No se encuentran las columnas necesarias ('Delay Time', 'Taper fecha_date', y/o 'Bottleneck') en los datos filtrados.")
else:
    st.warning("No hay datos que coincidan con los filtros seleccionados para graficar la tendencia de 'Delay Time'.")


filtered_df = filtered_df.reset_index(drop=True)

# Sección de Dashboard de Análisis-----------------------------------------------------------------------------------
st.markdown('<div id="datos-filtrados"></div>', unsafe_allow_html=True)
st.title("Datos Filtrados")
st.dataframe(filtered_df)
