from api import app
import base64
import hashlib
import io
from pathlib import Path

import cv2
import pandas as pd
import requests
import streamlit as st
from PIL import Image


API= "http://127.0.0.1:8000"
PASTA_EXEMPLOS = Path(__file__).parent
INTERVALO = 2

CAMERAS  = {
    "Professor": "http://192.168.0.10:8001/frame",
    "Bancada 1": "http://192.168.0.11:8001/frame",
    "Celular 1": "http://192.168.0.21:8001/shot.jpg",
}
EMOJIS = {
    "feliz": "\U0001F604",
    "surpresa": "\U0001F632",
    "neutro": "\U0001F610", 
    "triste": "\U0001F622", 
    "raiva": "\U0001F620", 
    "medo": "\U0001F628", 
    "nojo": "\U0001F922", 
}
CSS = """
<style>
.quadro { position: relative; display: inline-block; }
.quadro img { max-width: 100%; border-radius: 8px; }
.rosto { position: absolute; border: 3px solid #F2C46D; border-radius: 6px; }
.emoji { position: absolute; font-size: 64px; transform: translate(-50%, -50%); animation: salto 0.6s ease-out; }
@keyframes salto {
0% { transform: translate(-50%, -50%) scale(0); }
60% { transform: translate(-50%, -110%) scale(1.3); }
100% { transform: translate(-50%, -50%) scale(1); }
}
</style>
"""
def preparar(imagem):
    imagem = imagem.convert("RGB")
    imagem.thumbnail((640, 640))
    return imagem
# Converte a imagem em bytes JPEG
def em_jpeg(imagem):
    buffer = io.BytesIO()
    imagem.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
def analisar(imagem, registrar=True):
    r = requests.post(f"{API}/analisar",
    params={"registrar": str(registrar).lower()},

    files={"imagem": ("quadro.jpg", em_jpeg(imagem), "image/jpeg")}, timeout=15)
    r.raise_for_status()
    return r.json()
# Imagem à esquerda, JSON da API à direita
def exibir(imagem, resposta, legenda):
    esquerda, direita = st.columns([3, 2])
    with esquerda:
        mostrar(imagem, resposta)
    st.caption(legenda)
    with direita:
        st.markdown("**Resposta da API**")
    st.json(resposta)
def mostrar(imagem, resposta):
    foto = base64.b64encode(em_jpeg(imagem)).decode()
    largura, altura = imagem.size
    pct = lambda valor, total: f"{100 * valor / total:.1f}%" # posição em % da imagem
    extras = ""
    for rosto in resposta["rostos"]:
        x, y, w, h = rosto["caixa"]
        extras += (
            f'<div class="rosto" style="left:{pct(x, largura)}; top:{pct(y, altura)};'
            f'width:{pct(w, largura)}; height:{pct(h, altura)}"></div>'
            f'<div class="emoji" style="left:{pct(x + w, largura)}; top:{pct(y,altura)}">'
            f'{EMOJIS[rosto["emocao"]]}</div>')
    st.html(CSS + f'<div class="quadro"><img src="data:image/jpeg;base64,{foto}">{extras}</div>')
# Ex.: "2 rostos: feliz 92%, neutro 70%"
def descrever(resposta):
    if not resposta["encontrou_rosto"]:
        return "Nenhum rosto encontrado. Tente com mais luz e de frente para a câmera."
    partes = [f"{r['emocao']} {r['confianca']:.0%}" for r in resposta["rostos"]]
    n = resposta["total_rostos"]
    return f"{n} rosto{'s' if n > 1 else ''}: " + ", ".join(partes)
def quadro_webcam():
    if "webcam" not in st.session_state:
        st.session_state.webcam = cv2.VideoCapture(0)
    cam = st.session_state.webcam
    for _ in range(4): # descarta quadros antigos que ficaram no buffer
        cam.grab()
    ok, quadro = cam.retrieve()
    return Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)) if ok else None
def quadro_da_sala(url):
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))
# Fecha a webcam para liberar o dispositivo
def liberar_webcam():
    cam = st.session_state.pop("webcam", None)
    if cam is not None:
        cam.release()
@st.fragment(run_every=INTERVALO) # este trecho roda sozinho a cada 2 segundos
def ao_vivo(camera):
    try:
        if camera == "Webcam deste computador":
            imagem = quadro_webcam()
        else:
            imagem = quadro_da_sala(CAMERAS[camera])
    except requests.RequestException as erro:
        st.error(f"Não foi possível acessar a câmera '{camera}': {erro}")
        return
    if imagem is None:
        st.error("Não consegui ler a webcam. Ela está conectada e livre?")
        return
    
    imagem = preparar(imagem)
    try:
        resposta = analisar(imagem) # a API analisa e já registra a avaliação
    except requests.RequestException as erro:
        st.error(f"Falha ao falar com a API: {erro}")
        return
    legenda = descrever(resposta)
    if resposta["encontrou_rosto"]:
        legenda += " · registrado na API"
    exibir(imagem, resposta, legenda)

def obter_foto():
    fonte = st.radio("Fonte da imagem", ["Câmera do computador", "Galeria de exemplos", "Upload"], horizontal=True)
    if fonte == "Câmera do computador":
        foto = st.camera_input("Tire uma foto")
        return (Image.open(foto), None) if foto else (None, None)
    if fonte == "Galeria de exemplos":
        arquivos = sorted(p for p in PASTA_EXEMPLOS.glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not arquivos:
            st.info("A pasta exemplos/ está vazia. Rode antes: python baixar_exemplos.py")
            return None, None
        arquivo = st.selectbox("Exemplo", arquivos, format_func=lambda p: p.name)
        esperado = arquivo.stem.split("_")[0] # ex.: feliz_01.jpg -> feliz
        return Image.open(arquivo), (esperado if esperado in EMOJIS else None)
    enviado = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    return (Image.open(enviado), None) if enviado else (None, None)

@st.fragment(run_every=5)
def resumo_da_turma():
    st.subheader("Resumo da turma")
    try:
        resumo = requests.get(f"{API}/resumo", timeout=5).json()
    except requests.ConnectionError:
        st.error("API fora do ar. Rode antes: uvicorn api:app --reload")
        return
    pol = resumo["polaridade"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", resumo["total"])
    c2.metric("Positivo", pol.get("positivo", 0))
    c3.metric("Neutro", pol.get("neutro", 0))
    c4.metric("Negativo", pol.get("negativo", 0))
    if resumo["emocoes"]:
        st.bar_chart(pd.Series(resumo["emocoes"], name="avaliações"))
st.title("\U0001F60A Termômetro da Aula")
modo = st.radio("Modo", ["Ao vivo", "Foto"], horizontal=True)
if modo == "Ao vivo":
    camera = st.selectbox("Câmera", ["Webcam deste computador", *CAMERAS])
    if camera != "Webcam deste computador":
        liberar_webcam()
    if st.toggle(f"Analisar a cada {INTERVALO} segundos"):
        ao_vivo(camera)
    else:
        liberar_webcam()
else:
    liberar_webcam()
    imagem, esperado = obter_foto()
    if imagem is not None:
        imagem = preparar(imagem)
        chave = hashlib.md5(em_jpeg(imagem)).hexdigest()
        nova = chave != st.session_state.get("ultima_foto")
        try:
            resposta = analisar(imagem, registrar=nova)
        except requests.RequestException as erro:
            st.error(f"Falha ao falar com a API: {erro}")
            st.stop()
        st.session_state.ultima_foto = chave
        legenda = descrever(resposta)
        if resposta["encontrou_rosto"]:
            legenda += " · registrado na API" if nova else " · foto já registrada"
            if esperado: 
                acerto = "\u2705" if esperado == resposta["rostos"][0]["emocao"] else "\u274C"
                legenda += f" · {acerto} esperado: {esperado}"
            exibir(imagem, resposta, legenda)
st.divider()
resumo_da_turma()