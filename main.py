"""
Scanner com YOLO — Demonstração de detecção de objetos via câmera usando Streamlit.
Protótipo leve, single-file, pronto para deploy no Render.
"""

from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Scanner com Yolo", layout="centered")
st.title("Scanner com Yolo")


# ---------------------------------------------------------------------------
# Carregamento do modelo YOLO (cacheado para evitar recarregar a cada rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo YOLO...")
def carregar_modelo(caminho_modelo: str = "yolov8n.pt") -> YOLO:
    """Carrega e retorna o modelo YOLO especificado."""
    try:
        return YOLO(caminho_modelo)
    except Exception as erro:
        st.error(f"Falha ao carregar o modelo YOLO: {erro}")
        st.stop()


modelo = carregar_modelo()


# ---------------------------------------------------------------------------
# Captura de imagem via câmera
# Observação: em ambientes de nuvem (Render) não há acesso direto a cv2.VideoCapture,
# pois o servidor não possui câmera física. O widget st.camera_input acessa a câmera
# do dispositivo do usuário através do navegador, o que é a abordagem correta para deploy.
# ---------------------------------------------------------------------------
imagem_capturada = st.camera_input("Abrir câmera e capturar imagem")


# ---------------------------------------------------------------------------
# Processamento e inferência sobre a imagem capturada
# ---------------------------------------------------------------------------
def detectar_objetos(imagem_pil: Image.Image, modelo_yolo: YOLO) -> Optional[np.ndarray]:
    """Executa a inferência YOLO sobre uma imagem PIL e retorna o frame anotado (RGB)."""
    try:
        imagem_np = np.array(imagem_pil.convert("RGB"))
        resultados = modelo_yolo.predict(source=imagem_np, verbose=False)
        frame_anotado_bgr = resultados[0].plot()
        frame_anotado_rgb = frame_anotado_bgr[:, :, ::-1]
        return frame_anotado_rgb
    except Exception as erro:
        st.error(f"Falha durante a detecção de objetos: {erro}")
        return None


if imagem_capturada is not None:
    imagem_pil = Image.open(imagem_capturada)
    frame_resultado = detectar_objetos(imagem_pil, modelo)

    if frame_resultado is not None:
        st.image(frame_resultado, caption="Objetos detectados", use_container_width=True)