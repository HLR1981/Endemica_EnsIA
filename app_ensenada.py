import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="EndémicaEns",
    page_icon="🌸",
    layout="centered"
)

# ==========================================
# CARGAR MODELO
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_flores_ensenada.keras", compile=False)

model = load_model()

# ==========================================
# CLASES Y DICCIONARIOS
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
    'Choya Californiana': "Cactus característico de zonas áridas y desérticas.",
    'Encelia Farinosa': "Arbusto resistente al calor con flores amarillas.",
    'Encino Quercus': "Árbol emblemático de ecosistemas mediterráneos.",
    'Lila de California': "Arbusto con flores azuladas y aroma agradable.",
    'Maguey de la Costa': "Planta suculenta endémica de Baja California.",
    'Rosa de Castilla': "Rosa silvestre poco común de la región.",
    'Salvia de Munz': "Planta aromática endémica con flores violetas."
}

# ==========================================
# DISEÑO VISUAL: ENCABEZADO
# ==========================================
# Un banner sutil de naturaleza para darle color al inicio
st.image("https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=1000&q=80", use_container_width=True)

st.title("🌸 EndémicaEns")
st.subheader("Identificador de flora endémica de Ensenada")
st.markdown("---")

# ==========================================
# DISEÑO EN COLUMNAS (AQUÍ ESTÁ EL TRUCO)
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Sube tu foto")
    st.write("Selecciona o toma una fotografía de cerca para que la IA la analice.")
    
    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed" # Ocultamos el texto repetitivo del uploader
    )
    
    if archivo is not None:
        img = Image.open(archivo).convert("RGB")
        st.image(img, caption="Imagen cargada con éxito", use_container_width=True)

with col2:
    st.markdown("### 🧠 Diagnóstico de la IA")
    
    if archivo is None:
        # Estado de espera amigable
        st.info("💡 **Sistema listo.** Esperando que subas una imagen en la sección de la izquierda para iniciar el reconocimiento.")
    else:
        # Procesamiento de la predicción
        img_resized = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        with st.spinner('Analizando características botánicas...'):
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            indice = np.argmax(score)

        carpeta_original = nombres_clases[indice]
        nombre_mostrar = nombres_limpios.get(carpeta_original, carpeta_original)
        confianza = 100 * np.max(score)

        # Despliegue de resultados con diseño limpio
        if confianza < 70:
            st.error("⚠️ **No se pudo identificar con seguridad.** La confianza es muy baja. Intenta tomar la foto con mejor iluminación o más cerca de la flor.")
        else:
            st.success(f"### Especie detectada:\n## **{nombre_mostrar}**")
            
            # Usamos st.metric para que el porcentaje se vea gigante y genial
            st.metric(label="Certeza del análisis", value=f"{confianza:.2f}%")
            st.progress(float(confianza) / 100)
            
            st.markdown("#### ℹ️ Descripción")
            descripcion = info_flores.get(nombre_mostrar, "Información no disponible.")
            st.info(descripcion)

# ==========================================
# SECCIÓN INFERIOR: DESGLOSE COMPLETO
# ==========================================
if archivo is not None:
    st.markdown("---")
    with st.expander("📊 Ver desglose técnico de probabilidades"):
        st.write("Este es el nivel de coincidencia que la inteligencia artificial asignó a cada una de las especies posibles:")
        for i, prob in enumerate(score):
            nombre = nombres_limpios.get(nombres_clases[i], nombres_clases[i])
            porcentaje = 100 * prob
            st.write(f"**{nombre}**: {porcentaje:.2f}%")
            st.progress(float(prob))
