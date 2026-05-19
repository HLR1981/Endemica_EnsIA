import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Pestaña del navegador)
# ==========================================
st.set_page_config(
    page_title="EndémicaEns",
    page_icon="🌸",
    layout="centered"
)

# ==========================================
# 2. CARGAR MODELO
# ==========================================
@st.cache_resource
def load_model():
    # Asegúrate de que el nombre coincida con tu archivo .keras en GitHub
    return tf.keras.models.load_model("modelo_flores_ensenada.keras", compile=False)

model = load_model()

# ==========================================
# 3. DICCIONARIOS DE DATOS
# ==========================================
nombres_clases = [
    'amapola_de_california', 'choya_californiana', 'encelia_farinosa',
    'encino_quercus_agrifolia', 'lila_california_ceanothus',
    'maguey_costa_agave_shawii', 'rosa_castilla_rosa_minutifolia', 'salvia_munzii'
]

nombres_limpios = {
    'amapola_de_california': 'Amapola de California',
    'choya_californiana': 'Choya Californiana',
    'encelia_farinosa': 'Encelia Farinosa',
    'encino_quercus_agrifolia': 'Encino Quercus',
    'lila_california_ceanothus': 'Lila de California',
    'maguey_costa_agave_shawii': 'Maguey de la Costa',
    'rosa_castilla_rosa_minutifolia': 'Rosa de Castilla',
    'salvia_munzii': 'Salvia de Munz'
}

info_flores = {
    'Amapola de California': "Flor silvestre de color naranja brillante, muy común en California y Baja California.",
    'Choya Californiana': "Cactus característico de zonas áridas y desérticas de la región.",
    'Encelia Farinosa': "Arbusto resistente al calor con flores amarillas, típica del matorral costero.",
    'Encino Quercus': "Árbol emblemático de los arroyos y zonas frescas de Ensenada.",
    'Lila de California': "Arbusto con flores azuladas y aroma agradable que embellece los cerros.",
    'Maguey de la Costa': "Planta suculenta majestuosa, endémica de las costas de Baja California.",
    'Rosa de Castilla': "Rosa silvestre nativa de la región, muy valorada por su rareza.",
    'Salvia de Munz': "Planta aromática endémica con flores violetas intensas."
}

# ==========================================
# 4. DISEÑO VISUAL: ENCABEZADO
# ==========================================
# Banner local (tu foto de la Carretera Escénica)
st.image(
    "banner.jpg.png", 
    use_container_width=True, 
    caption="Carretera Escénica Tijuana-Ensenada, Baja California"
)

# TÍTULO ÚNICO (Solo uno aquí abajo)
st.title("🌸 EndémicaEns")
st.markdown("### **Identificador Inteligente de Flora Endémica de Ensenada**")
st.write("Identifica las plantas nativas de nuestra región usando Inteligencia Artificial. Sube una foto de cerca para un mejor diagnóstico.")
st.markdown("---")

# ==========================================
# 5. DISEÑO EN COLUMNAS
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Sube tu foto")
    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if archivo is not None:
        img = Image.open(archivo).convert("RGB")
        st.image(img, caption="Imagen cargada para análisis", use_container_width=True)

with col2:
    st.markdown("### 🧠 Diagnóstico de la IA")
    
    if archivo is None:
        st.info("💡 **Sistema listo.** Sube una foto a la izquierda para identificar la especie.")
    else:
        # Procesamiento
        img_resized = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        with st.spinner('Analizando botánica...'):
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            indice = np.argmax(score)

        nombre_mostrar = nombres_limpios.get(nombres_clases[indice], nombres_clases[indice])
        confianza = 100 * np.max(score)

        # Resultados
        if confianza < 70:
            st.error("⚠️ **Confianza baja.** No estoy seguro de qué planta es. Intenta con una foto más clara.")
        else:
            st.success(f"## **{nombre_mostrar}**")
            st.metric(label="Certeza", value=f"{confianza:.2f}%")
            st.progress(float(confianza) / 100)
            
            st.markdown("#### ℹ️ Sobre esta especie")
            st.info(info_flores.get(nombre_mostrar, "Información no disponible."))

# ==========================================
# 6. SECCIÓN INFERIOR
# ==========================================
if archivo is not None:
    st.markdown("---")
    with st.expander("📊 Ver análisis técnico detallado"):
        for i, prob in enumerate(score):
            nombre = nombres_limpios.get(nombres_clases[i], nombres_clases[i])
            st.write(f"**{nombre}**: {100 * prob:.2f}%")
            st.progress(float(prob))
