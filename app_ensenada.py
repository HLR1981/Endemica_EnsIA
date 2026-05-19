import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd  # Se añade para estructurar los datos del mapa

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
    'Amapola de California': "Flor silvestre de color naranja brillante, muy común en laderas y caminos de Baja California.",
    'Choya Californiana': "Cactus característico de las zonas arbustivas áridas y valles de la región.",
    'Encelia Farinosa': "Arbusto resistente al calor con flores amarillas, común en zonas de matorral costero.",
    'Encino Quercus': "Árbol emblemático de los ecosistemas mediterráneos y cañadas de Ensenada.",
    'Lila de California': "Arbusto con densas flores azuladas, muy visto en las zonas de chaparral.",
    'Maguey de la Costa': "Planta suculenta espectacular y endémica que crece junto a los acantilados marinos.",
    'Rosa de Castilla': "Rosa silvestre nativa de la región, sumamente valorada por su rareza y belleza.",
    'Salvia de Munz': "Planta arbustiva aromática endémica con llamativas flores violetas."
}

# Coordenadas estimadas de avistamiento común en el municipio de Ensenada
coordenadas_flores = {
    'Amapola de California': {"lat": 31.8800, "lon": -116.6000},       # Laderas del Cañón de Doña Petra
    'Choya Californiana': {"lat": 31.9500, "lon": -116.4500},          # Zonas semiáridas rumbo a Ojos Negros
    'Encelia Farinosa': {"lat": 31.7500, "lon": -116.6000},            # Cerros cercanos a Maneadero
    'Encino Quercus': {"lat": 32.0500, "lon": -116.6000},              # Cañadas en el Valle de Guadalupe
    'Lila de California': {"lat": 31.0000, "lon": -115.6000},          # Alrededores de la Sierra de San Pedro Mártir
    'Maguey de la Costa': {"lat": 31.7144, "lon": -116.7197},          # Acantilados cerca de La Bufadora
    'Rosa de Castilla': {"lat": 31.6500, "lon": -116.4000},            # Valles interiores de la antigua ruta costera
    'Salvia de Munz': {"lat": 31.8600, "lon": -116.6500}               # Chaparral en los cerros detrás de la ciudad
}

# ==========================================
# DISEÑO VISUAL: ENCABEZADO
# ==========================================
st.image("https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=1000&q=80", use_container_width=True)

st.title("🌸 EndémicaEns")
st.subheader("Identificador de flora endémica de Ensenada")
st.markdown("---")

# ==========================================
# DISEÑO EN COLUMNAS
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Captura o Sube tu foto")
    
    # MEJORA 2: Cámara en vivo integrada
    foto_camara = st.camera_input("Toma una foto en vivo desde tu dispositivo")
    
    st.markdown("**O selecciona una imagen de tu galería:**")
    archivo_subido = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Decidir cuál recurso utilizar (prioriza la cámara si se tomó una foto)
    archivo = None
    if foto_camara is not None:
        archivo = foto_camara
    elif archivo_subido is not None:
        archivo = archivo_subido

    if archivo is not None and foto_camara is None:
        img = Image.open(archivo).convert("RGB")
        st.image(img, caption="Imagen cargada con éxito", use_container_width=True)

with col2:
    st.markdown("### 🧠 Diagnóstico de la IA")
    
    if archivo is None:
        st.info("💡 **Sistema listo.** Captura una foto en vivo o sube una imagen a la izquierda para iniciar el reconocimiento.")
    else:
        if foto_camara is not None:
            img = Image.open(archivo).convert("RGB")
            
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

        # Despliegue de resultados
        if confianza < 70:
            st.error("⚠️ **No se pudo identificar con seguridad.** La confianza es muy baja. Intenta tomar la foto con mejor iluminación o más cerca de la flor.")
        else:
            st.success(f"### Especie detectada:\n## **{nombre_mostrar}**")
            
            st.metric(label="Certeza del análisis", value=f"{confianza:.2f}%")
            st.progress(float(confianza) / 100)
            
            st.markdown("#### ℹ️ Sobre esta especie")
            descripcion = info_flores.get(nombre_mostrar, "Información no disponible.")
            st.info(descripcion)
            
            # MEJORA 1: Mapa dinámico interactivo según la planta detectada
            st.markdown("#### 🗺️ Zona de Avistamiento Estimada")
            if nombre_mostrar in coordenadas_flores:
                coordenadas = coordenadas_flores[nombre_mostrar]
                # Creamos el DataFrame que Streamlit necesita para dibujar los puntos del mapa
                data_mapa = pd.DataFrame([{
                    'lat': coordenadas['lat'],
                    'lon': coordenadas['lon']
                }])
                st.write(f"Zonas comunes de muestreo para: *{nombre_mostrar}* en Ensenada.")
                st.map(data_mapa, zoom=10)

# ==========================================
# SECCIÓN INFERIOR: DESGLOSE COMPLETO
# ==========================================
if archivo is not None and confianza >= 70:
    st.markdown("---")
    with st.expander("📊 Ver desglose técnico de probabilidades"):
        st.write("Este es el nivel de coincidencia que la inteligencia artificial asignó a cada una de las especies posibles:")
        for i, prob in enumerate(score):
            nombre = nombres_limpios.get(nombres_clases[i], nombres_clases[i])
            porcentaje = 100 * prob
            st.write(f"**{nombre}**: {porcentaje:.2f}%")
            st.progress(float(prob))
