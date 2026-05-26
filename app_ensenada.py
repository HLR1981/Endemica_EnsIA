import streamlit as tf_st  # Evita conflictos si se usa de forma cruzada
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# 1. CONFIGURACIÓN DE LA PÁGINA (Debe ser la primera línea de Streamlit)
st.set_page_config(
    page_title="EndémicaEns - Identificador de Flora",
    page_icon="🌸",
    layout="wide"
)

# 2. CARGAR EL MODELO DE IA (Con caché para que no sea lento)
@st.cache_resource
def cargar_modelo_ia():
    # Carga tu modelo entrenado sin compilar para ahorrar memoria en el servidor
    return tf.keras.models.load_model("modelo_flores_ensenada.keras", compile=False)

modelo = cargar_modelo_ia()

# Lista estricta de las 8 especies de tu dataset (Asegúrate de que coincida con tu orden de entrenamiento)
CLASES_BOTANICAS = [
    "Amapola de California",
    "Choya Californiana",
    "Encelia Farinosa",
    "Encino Quercus",
    "Lila de California",
    "Maguey de la Costa",
    "Rosa de Castilla",
    "Salvia de Munz"
]

# Diccionario de información complementaria y coordenadas estimadas en Ensenada
INFORMACION_PLANTAS = {
    "Amapola de California": {
        "desc": "Flor de color naranja brillante, común en valles y laderas soleadas. Muy representativa de la región.",
        "lat": 31.8900, "lon": -116.5900  # Cañón de Doña Petra
    },
    "Choya Californiana": {
        "desc": "Cactus característico de las zonas arbustivas áridas y valles de la región.",
        "lat": 32.0100, "lon": -116.4500  # Valle de Guadalupe / Zonas altas
    },
    "Encelia Farinosa": {
        "desc": "Arbusto resistente al calor con flores amarillas, común en zonas de matorral costero.",
        "lat": 31.8600, "lon": -116.6500  # Lomas de San Miguel
    },
    "Encino Quercus": {
        "desc": "Árbol nativo de gran tamaño, vital para los arroyos y cañadas de Ensenada.",
        "lat": 31.7500, "lon": -116.4800  # San Antonio de las Minas
    },
    "Lila de California": {
        "desc": "Arbusto con densos racimos de flores azules o moradas que florece en primavera en el chaparral.",
        "lat": 31.9500, "lon": -116.5000  # Sierra de Juárez (Zonas bajas)
    },
    "Maguey de la Costa": {
        "desc": "Agave endémico que crece pegado a los acantilados marinos de la costa del Pacífico.",
        "lat": 31.7300, "lon": -116.7100  # La Bufadora / Punta Banda
    },
    "Rosa de Castilla": {
        "desc": "Arbusto espinoso con flores rosas de alto valor endémico en el matorral ensenadense.",
        "lat": 31.6500, "lon": -116.5500  # El Rosario / Zonas del sur
    },
    "Salvia de Munz": {
        "desc": "Planta aromática con flores azules, endémica de la zona fronteriza y el matorral de Ensenada.",
        "lat": 31.8200, "lon": -116.6000  # El Estero / Maneadero
    }
}

# 3. FUNCIÓN INTELIGENTE DE PREPROCESAMIENTO Y ACLARADO AUTOMÁTICO
def procesar_y_aclarar_imagen(imagen_pil):
    """
    Analiza los píxeles de la imagen. Si está muy oscura, aumenta el brillo
    y contraste de forma matemática antes de pasarla por la IA.
    """
    # Convertir temporalmente a escala de grises para medir la luz (0 = negro total, 255 = blanco)
    imagen_gris = imagen_pil.convert('L')
    matriz_pixeles = np.array(imagen_gris)
    brillo_promedio = matriz_pixeles.mean()
    
    # Si el brillo es menor a 85, la foto necesita luz artificial virtual
    if brillo_promedio < 85:
        # Ajustamos el factor dependiendo de qué tan oscura esté la foto
        factor_brillo = 1.6 if brillo_promedio < 50 else 1.3
        
        # Aplicar el potenciador de brillo
        realzador_brillo = ImageEnhance.Brightness(imagen_pil)
        imagen_pil = realzador_brillo.enhance(factor_brillo)
        
        # Aplicar un toque de contraste para rescatar bordes y siluetas de las hojas
        realzador_contraste = ImageEnhance.Contrast(imagen_pil)
        imagen_pil = realzador_contraste.enhance(1.2)
        
    # Redimensionar al estándar que exige tu modelo entrenado (224x224)
    imagen_redimensionada = imagen_pil.resize((224, 224))
    
    # Convertir a matriz numérica (tensor) y expandir el lote (batch)
    arreglo_imagen = tf.keras.utils.img_to_array(imagen_redimensionada)
    arreglo_imagen = tf.expand_dims(arreglo_imagen, 0)
    
    return arreglo_imagen

# 4. INTERFAZ GRÁFICA DE USUARIO (UI)
st.image("https://images.unsplash.com/photo-1500627869374-13cd993b1115?auto=format&fit=crop&w=1200&q=80", use_container_width=True)

st.title("🌸 EndémicaEns")
st.subheader("Identificador de flora endémica de Ensenada")
st.write("Sube una foto de tu galería o usa la cámara de tu dispositivo móvil. El sistema aclarará la imagen de forma automática si detecta falta de luz.")

st.markdown("---")

# Crear el diseño de dos columnas paralelas
col_izquierda, col_derecha = st.columns(2)

with col_izquierda:
    st.markdown("### 📸 Captura o Sube tu foto")
    
    # Entrada 1: Cámara en vivo
    foto_camara = st.camera_input("Toma una foto en vivo desde tu dispositivo")
    
    # Entrada 2: Galería de imágenes
    foto_archivo = st.file_uploader("O selecciona una imagen de tu galería:", type=["jpg", "jpeg", "png"])
    
    # Determinar cuál de las dos entradas se va a procesar prioritariamente
    imagen_final = None
    if foto_camara is not None:
        imagen_final = foto_camara
    elif foto_archivo is not None:
        imagen_final = foto_archivo

with col_derecha:
    st.markdown("### 🧠 Diagnóstico de la IA")
    
    if imagen_final is not None:
        with st.spinner("Analizando y optimizando imagen..."):
            # 1. Abrir la imagen cargada con PIL
            imagen_pil = Image.open(imagen_final)
            
            # 2. Pasar la imagen por el filtro de brillo y conversión a matriz
            matriz_preparada = procesar_y_aclarar_imagen(imagen_pil)
            
            # 3. Realizar la predicción matemática
            predicciones = modelo.predict(matriz_preparada)
            puntuaciones_softmax = tf.nn.softmax(predicciones[0])
            
            # Indices de la predicción más alta
            indice_ganador = np.argmax(puntuaciones_softmax)
            nombre_especie = CLASES_BOTANICAS[indice_ganador]
            porcentaje_certeza = 100 * np.max(puntuaciones_softmax)
            
            # 4. Validar contra el umbral de seguridad (70%)
            if porcentaje_certeza >= 70.0:
                # Mostrar resultado exitoso
                st.success(f"### Especie detectada:\n## **{nombre_especie}**")
                st.metric(label="Certeza del análisis", value=f"{porcentaje_certeza:.2f}%")
                st.progress(int(porcentaje_certeza))
                
                # Desplegar información taxonómica y mapas si existe en el diccionario
                if nombre_especie in INFORMACION_PLANTAS:
                    info = INFORMACION_PLANTAS[nombre_especie]
                    
                    st.markdown("---")
                    st.markdown("### ℹ️ Sobre esta especie")
                    st.info(info["desc"])
                    
                    st.markdown("### 🗺️ Zona de Avistamiento Estimada")
                    st.write(f"Zonas comunes de muestreo para: *{nombre_especie}* en Ensenada.")
                    
                    # Estructurar las coordenadas en un DataFrame de Pandas para st.map
                    datos_mapa = pd.DataFrame({
                        'lat': [info["lat"]],
                        'lon': [info["lon"]]
                    })
                    st.map(datos_mapa, zoom=10)
            else:
                # Si la certeza es muy baja por mala calidad de imagen
                st.error("### No se pudo identificar con certeza")
                st.warning(f"La IA detectó similitud con **{nombre_especie}** ({porcentaje_certeza:.2f}%), pero no supera el umbral mínimo del 70% de seguridad. Por favor, intenta tomar otra foto con mejor iluminación, enfocando más de cerca las hojas o flores.")
    else:
        st.info("Esperando una imagen para iniciar el diagnóstico inteligente.")

# Código QR inferior dinámico para compartir la app
st.markdown("---")
col_qr1, col_qr2 = st.columns([1, 2])
with col_qr1:
    url_app = "https://endemicaensia-q8dp52x98awulusqgctonu.streamlit.app/"
    api_qr = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={url_app}"
    st.image(api_qr, caption="¡Escanea y comparte la app!", width=140)
with col_qr2:
    st.markdown("### 📲 ¡Lleva EndémicaEns contigo!")
    st.write("Escanea este código QR directamente con tu celular para abrir la app cuando estés haciendo trabajo de campo o senderismo por los cerros de Ensenada.")
