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
    # El modelo se carga sin compilar para evitar choques de versiones
    return tf.keras.models.load_model("modelo_flores_ensenada.keras", compile=False)

model = load_model()

# ==========================================
# CLASES (Manual para evitar FileNotFoundError)
# ==========================================
# Definimos la lista manualmente porque la carpeta 'dataset/train' no está en la nube
nombres_clases = [
    'amapola_de_california',
    'choya_californiana',
    'encelia_farinosa',
    'encino_quercus_agrifolia',
    'lila_california_ceanothus',
    'maguey_costa_agave_shawii',
    'rosa_castilla_rosa_minutifolia',
    'salvia_munzii'
]

# ==========================================
# NOMBRES BONITOS
# ==========================================
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

# ==========================================
# INFORMACIÓN DE FLORES
# ==========================================
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
# TÍTULO
# ==========================================
st.title("🌸 EndémicaEns")
st.subheader("Identificador de flora endémica de Ensenada")
st.write("Sube una imagen de una flor para identificarla.")

# ==========================================
# SUBIR IMAGEN
# ==========================================
archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICCIÓN
# ==========================================
if archivo is not None:
    # Procesamiento de imagen
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Redimensionar y preparar para el modelo
    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # Ejecutar predicción
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    indice = np.argmax(score)

    # Obtener nombres
    carpeta_original = nombres_clases[indice]
    nombre_mostrar = nombres_limpios.get(carpeta_original, carpeta_original)
    confianza = 100 * np.max(score)

    st.divider()

    # Validación de confianza
    if confianza < 70:
        st.error("⚠️ No puedo identificar esta flor con suficiente seguridad.")
    else:
        st.success(f"🌸 Resultado: {nombre_mostrar}")
        st.write(f"**Confianza:** {confianza:.2f}%")
        st.progress(float(confianza) / 100)

        descripcion = info_flores.get(nombre_mostrar, "Información no disponible.")
        st.info(descripcion)

    # Mostrar todas las probabilidades
    with st.expander("📊 Ver desglose de probabilidades"):
        for i, prob in enumerate(score):
            nombre = nombres_limpios.get(nombres_clases[i], nombres_clases[i])
            st.write(f"**{nombre}:** {100 * prob:.2f}%")
