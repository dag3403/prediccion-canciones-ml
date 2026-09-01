import os
import json
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard de Modelos ML", layout="wide")

st.title("📊 Dashboard Interactivo de Modelos de Machine Learning")
st.markdown("---")

# Mapeo de números de estrategia a nombres descriptivos
MAPEO_ESTRATEGIAS = {
    "1": "Sin balancear",
    "2": "Penalizando errores en las clases minoritarias",
    "3": "Balanceando a la mediana",
    "4": "Balanceando al percentil 75"
}

# Orden lógico para mostrar las estrategias ordenadas
ORDEN_ESTRATEGIAS = [
    "Sin balancear",
    "Penalizando errores en las clases minoritarias",
    "Balanceando a la mediana",
    "Balanceando al percentil 75"
]

# Función para cargar y parsear todos los JSON de la carpeta 'models'
@st.cache_data
def cargar_datos():
    data = []
    directorio = "models"  # Carpeta 'models' en la raíz del proyecto
    
    # Comprobar si la carpeta existe para evitar errores
    if not os.path.exists(directorio):
        st.error(f"⚠️ No se encontró la carpeta '{directorio}' en el directorio raíz.")
        return pd.DataFrame()
    
    for archivo in os.listdir(directorio):
        if archivo.endswith(".json") and archivo.startswith("resultados_"):
            # Ejemplos esperados: resultados_catboost_1.json, resultados_mlp_2.json, etc.
            nombre_limpio = archivo.replace("resultados_", "").replace(".json", "")
            partes = nombre_limpio.split("_")
            
            if len(partes) >= 2:
                # El modelo es la primera parte (ej. catboost, xgboost, mlp, lightgbm, knn)
                modelo = partes[0].upper()
                # La estrategia es la última parte (ej. 1, 2, 3, 4)
                estrategia_num = partes[-1]
                
                # Traducir el número de estrategia a su nombre descriptivo
                estrategia_nombre = MAPEO_ESTRATEGIAS.get(estrategia_num, f"Estrategia {estrategia_num}")
                
                ruta_completa = os.path.join(directorio, archivo)
                try:
                    with open(ruta_completa, "r", encoding="utf-8") as f:
                        contenido = json.load(f)
                        
                        metrics = contenido.get("metrics", {})
                        fila = {
                            "archivo": archivo,
                            "modelo": modelo,
                            "estrategia_num": estrategia_num,
                            "estrategia": estrategia_nombre,
                            "accuracy": metrics.get("accuracy"),
                            "precision_weighted": metrics.get("precision_weighted"),
                            "recall_weighted": metrics.get("recall_weighted"),
                            "f1_score_weighted": metrics.get("f1_score_weighted"),
                            "roc_auc_weighted": metrics.get("roc_auc_weighted"),
                            "best_parameters": contenido.get("best_parameters", {}),
                            "confusion_matrix": contenido.get("confusion_matrix", [])
                        }
                        data.append(fila)
                except Exception as e:
                    st.error(f"Error leyendo {archivo}: {e}")
                    
    df_temp = pd.DataFrame(data)
    if not df_temp.empty:
        # Categorizar para ordenar correctamente las estrategias
        df_temp["estrategia"] = pd.Categorical(df_temp["estrategia"], categories=ORDEN_ESTRATEGIAS, ordered=True)
    return df_temp

df = cargar_datos()

if df.empty:
    st.warning("⚠️ No se encontraron archivos JSON válidos dentro de la carpeta 'models'.")
else:
    # --- BARRA LATERAL (Filtros) ---
    st.sidebar.header("🔍 Configuración de Vista")
    
    modo_vista = st.sidebar.radio(
        "Selecciona el modo de visualización:", 
        [
            "Vista Independiente (Por Modelo y Estrategia)", 
            "Comparativa de las 4 estrategias (Por Modelo)", 
            "Vista Global (Comparativa de Todos)"
        ]
    )
    
    modelos_disponibles = sorted(df["modelo"].unique())
    
    if modo_vista == "Vista Independiente (Por Modelo y Estrategia)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Controles de Selección")
        
        # 1. Selector de Modelo
        modelo_seleccionado = st.sidebar.selectbox("Selecciona el Modelo:", modelos_disponibles)
        
        # Filtrar el DataFrame por el modelo elegido
        df_modelo = df[df["modelo"] == modelo_seleccionado]
        
        # 2. Selector de Estrategia de Balanceo dinámico
        estrategias_disponibles = [e for e in ORDEN_ESTRATEGIAS if e in df_modelo["estrategia"].values]
        estrategia_seleccionada = st.sidebar.selectbox("Selecciona la Estrategia de Balanceo:", estrategias_disponibles)
        
        # Filtrar el registro exacto
        row = df_modelo[df_modelo["estrategia"] == estrategia_seleccionada].iloc[0]
        
        st.header(f"🤖 Modelo: `{modelo_seleccionado}` | ⚖️ {estrategia_seleccionada}")
        st.caption(f"Archivo origen: models/{row['archivo']}")
        
        # Mostrar métricas y matriz de manera limpia e independiente
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Métricas de Rendimiento")
            m_df = pd.DataFrame({
                "Métrica": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)", "ROC AUC (Weighted)"],
                "Valor": [
                    row["accuracy"], 
                    row["precision_weighted"], 
                    row["recall_weighted"], 
                    row["f1_score_weighted"], 
                    row["roc_auc_weighted"]
                ]
            })
            st.dataframe(m_df.set_index("Métrica"), use_container_width=True)
            
            st.subheader("⚙️ Mejores Parámetros (Best Parameters)")
            st.json(row["best_parameters"])
        
        with col2:
            st.subheader("🧩 Matriz de Confusión")
            cm = row["confusion_matrix"]
            if cm:
                fig, ax = plt.subplots(figsize=(6, 5))
                
                # --- MEJORA VISUAL DE LA MATRIZ DE CONFUSIÓN ---
                sns.heatmap(
                    cm, 
                    annot=True,         # Muestra los números dentro de las celdas
                    fmt="d",            # Formato entero para las cantidades
                    cmap="Blues",       # Escala de azules con alto contraste
                    cbar=True,          # Barra de color lateral
                    linewidths=1.5,     # Grosor de la línea de la cuadrícula
                    linecolor='white',  # Color blanco para separar claramente los cuadrados
                    ax=ax
                )
                
                ax.set_xlabel("Predicción", fontsize=11, fontweight='bold')
                ax.set_ylabel("Valor Real", fontsize=11, fontweight='bold')
                st.pyplot(fig)
            else:
                st.info("No hay matriz de confusión disponible en este archivo.")
                
    elif modo_vista == "Comparativa de las 4 estrategias (Por Modelo)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Configuración")
        
        # Selector de un único modelo para ver sus 4 opciones juntas
        modelo_seleccionado = st.sidebar.selectbox("Selecciona el Modelo a analizar:", modelos_disponibles)
        
        st.header(f"⚖️ Comparativa de Estrategias para el Modelo: `{modelo_seleccionado}`")
        
        df_modelo = df[df["modelo"] == modelo_seleccionado].sort_values("estrategia")
        
        if df_modelo.empty:
            st.warning("No hay datos suficientes para este modelo.")
        else:
            # Gráfico de barras comparativo para este modelo específico
            metrica_grafico = st.selectbox(
                "Selecciona la métrica a visualizar en el gráfico:",
                ["f1_score_weighted", "accuracy", "precision_weighted", "recall_weighted", "roc_auc_weighted"]
            )
            
            st.subheader(f"Rendimiento por estrategia ({metrica_grafico.upper()})")
            st.bar_chart(df_modelo.set_index("estrategia")[metrica_grafico])
            
            st.markdown("---")
            st.subheader("📋 Tabla Comparativa Detallada")
            columnas_mostrar = ["estrategia", "accuracy", "precision_weighted", "recall_weighted", "f1_score_weighted", "roc_auc_weighted", "archivo"]
            st.dataframe(df_modelo[columnas_mostrar].set_index("estrategia"), use_container_width=True)
                
    else:
        # --- VISTA GLOBAL / COMPARATIVA ---
        st.header("📈 Comparativa Global de Todos los Modelos y Estrategias")
        
        # Filtro opcional por modelo en la vista global
        modelos_seleccionados = st.sidebar.multiselect(
            "Filtrar modelos a comparar:", 
            modelos_disponibles, 
            default=modelos_disponibles
        )
        
        df_global = df[df["modelo"].isin(modelos_seleccionados)]
        
        metrica_a_comparar = st.selectbox(
            "Selecciona la métrica principal para comparar:",
            ["f1_score_weighted", "accuracy", "precision_weighted", "recall_weighted", "roc_auc_weighted"]
        )
        
        if not df_global.empty:
            st.subheader(f"Gráfico comparativo de {metrica_a_comparar.upper()}")
            # Crear tabla pivote para visualizar barras agrupadas por estrategia y modelo
            pivote = df_global.pivot(index="estrategia", columns="modelo", values=metrica_a_comparar)
            st.bar_chart(pivote)
        
        st.markdown("---")
        st.subheader("📊 Tabla Resumen Consolidada")
        columnas_mostrar = ["modelo", "estrategia", "accuracy", "precision_weighted", "recall_weighted", "f1_score_weighted", "roc_auc_weighted", "archivo"]
        st.dataframe(df_global[columnas_mostrar], use_container_width=True)