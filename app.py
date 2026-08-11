"""
Scanner com YOLO — Detecção de objetos em tempo real via câmera usando Streamlit.
Protótipo leve, single-file, pronto para deploy no Render.

Observação de arquitetura: em nuvem (Render) o servidor não possui câmera física, então
cv2.VideoCapture(0) não é viável. Para tempo real, a câmera do dispositivo do usuário é
acessada pelo navegador via WebRTC (streamlit-webrtc) e cada frame é enviado ao servidor,
onde o YOLO roda a inferência antes de o frame anotado retornar ao navegador.
"""

import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


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
# Processador de vídeo: recebe cada frame do stream e aplica a inferência YOLO
# ---------------------------------------------------------------------------
class ProcessadorYolo(VideoProcessorBase):
    """Executa detecção de objetos em cada frame recebido da câmera do usuário."""

    def __init__(self) -> None:
        self.modelo = modelo

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            imagem_bgr = frame.to_ndarray(format="bgr24")
            resultados = self.modelo.predict(source=imagem_bgr, verbose=False)
            imagem_anotada = resultados[0].plot()
            return av.VideoFrame.from_ndarray(imagem_anotada, format="bgr24")
        except Exception:
            # Em caso de falha na inferência de um frame, devolve o frame original
            # para não interromper o streaming.
            return frame


# ---------------------------------------------------------------------------
# Botão/controle para abrir a câmera e iniciar a detecção em tempo real
# webrtc_streamer já fornece nativamente o botão de "Start"/"Stop" da câmera.
# ---------------------------------------------------------------------------
webrtc_streamer(
    key="scanner-yolo",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=ProcessadorYolo,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
