import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import zipfile
import os

# ─────────────────────────────────────────────
#  ⚙️  CONFIGURA AQUÍ TU MODELO
# ─────────────────────────────────────────────
CLASES = ["Sano", "Enfermedad A", "Enfermedad B"]   # <-- Cambia por tus clases reales
IMG_SIZE = 224                                        # <-- Tamaño de entrada de tu modelo

def cargar_modelo(ruta_pth):
    """Carga tu modelo .pth — ajusta la arquitectura si usaste otra."""
    modelo = models.resnet50(pretrained=False)         # <-- Cambia resnet50 si usaste otra
    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASES))
    estado = torch.load(ruta_pth, map_location="cpu")
    # Maneja si guardaste solo los pesos o el modelo completo
    if isinstance(estado, dict) and "state_dict" in estado:
        modelo.load_state_dict(estado["state_dict"])
    elif isinstance(estado, dict):
        modelo.load_state_dict(estado)
    else:
        modelo = estado
    modelo.eval()
    return modelo
# ─────────────────────────────────────────────

# Preprocesamiento estándar ImageNet
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predecir(modelo, img_pil):
    tensor = transform(img_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        salida = modelo(tensor)
        probs = torch.softmax(salida, dim=1)[0]
        idx = probs.argmax().item()
    return CLASES[idx], round(float(probs[idx]) * 100, 1), {c: round(float(p)*100,1) for c, p in zip(CLASES, probs)}


# ── UI ───────────────────────────────────────
st.set_page_config(page_title="Diagnóstico por Imagen", page_icon="🔬", layout="wide")

st.markdown("""
    <h1 style='text-align:center;'>🔬 Diagnóstico de Enfermedades por Imagen</h1>
    <p style='text-align:center; color:gray;'>Sube hasta 500 imágenes y obtén el diagnóstico automático con tu modelo entrenado</p>
    <hr>
""", unsafe_allow_html=True)

# ── Sidebar: cargar modelo ───────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_file = st.file_uploader("1. Sube tu modelo (.pth)", type=["pth", "pt"])
    st.markdown("---")
    st.info("2. Sube tus imágenes abajo (hasta 500)")
    st.markdown("---")
    st.caption("💡 Ajusta `CLASES` y la arquitectura al inicio del archivo `app.py`")

# ── Carga del modelo ─────────────────────────
modelo = None
if modelo_file:
    with st.spinner("Cargando modelo..."):
        ruta_tmp = "/tmp/modelo_cargado.pth"
        with open(ruta_tmp, "wb") as f:
            f.write(modelo_file.read())
        try:
            modelo = cargar_modelo(ruta_tmp)
            st.sidebar.success("✅ Modelo cargado correctamente")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar el modelo:\n{e}")

# ── Carga de imágenes ────────────────────────
st.subheader("📁 Sube tus imágenes")
archivos = st.file_uploader(
    "Selecciona hasta 500 imágenes (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if len(archivos) > 500:
    st.warning("⚠️ Se tomarán solo las primeras 500 imágenes.")
    archivos = archivos[:500]

# ── Botón de análisis ────────────────────────
if archivos and modelo:
    if st.button("🚀 Analizar imágenes", type="primary", use_container_width=True):

        resultados = []
        imagenes   = []
        progress   = st.progress(0, text="Analizando...")

        for i, archivo in enumerate(archivos):
            img = Image.open(io.BytesIO(archivo.read()))
            clase, confianza, probs_dict = predecir(modelo, img)
            resultados.append({
                "Imagen": archivo.name,
                "Diagnóstico": clase,
                "Confianza (%)": confianza,
                **{f"P({c}) %": v for c, v in probs_dict.items()}
            })
            imagenes.append((archivo.name, img, clase, confianza))
            progress.progress((i + 1) / len(archivos),
                              text=f"Procesando {i+1}/{len(archivos)}: {archivo.name}")

        progress.empty()
        df = pd.DataFrame(resultados)
        conteo = Counter(df["Diagnóstico"])

        # ── Métricas rápidas ──────────────────
        st.markdown("---")
        st.subheader("📊 Resumen")
        cols = st.columns(len(CLASES) + 1)
        cols[0].metric("Total imágenes", len(df))
        for i, clase in enumerate(CLASES):
            cols[i+1].metric(clase, conteo.get(clase, 0))

        # ── Gráficas ──────────────────────────
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🍕 Distribución (Pastel)")
            fig_pie = px.pie(
                values=list(conteo.values()),
                names=list(conteo.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.35,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("📊 Frecuencia por diagnóstico (Barras)")
            fig_bar = px.bar(
                x=list(conteo.keys()),
                y=list(conteo.values()),
                color=list(conteo.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"x": "Diagnóstico", "y": "Cantidad"},
                text_auto=True,
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Galería con diagnóstico ───────────
        st.markdown("---")
        st.subheader("🖼️ Galería de imágenes con diagnóstico")

        COLOR = {c: col for c, col in zip(
            CLASES, ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#3498db"]
        )}

        n_cols = 4
        chunk  = [imagenes[i:i+n_cols] for i in range(0, len(imagenes), n_cols)]

        for fila in chunk:
            cols_gal = st.columns(n_cols)
            for j, (nombre, img, clase, conf) in enumerate(fila):
                with cols_gal[j]:
                    st.image(img, use_container_width=True)
                    color = COLOR.get(clase, "#888")
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<span style='background:{color};color:white;padding:2px 8px;"
                        f"border-radius:12px;font-size:0.8em;'>{clase}</span><br>"
                        f"<small style='color:gray;'>{conf}% confianza</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # ── Tabla descargable ─────────────────
        st.markdown("---")
        st.subheader("📋 Tabla de resultados")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados CSV",
            data=csv,
            file_name="resultados_diagnostico.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif archivos and not modelo:
    st.info("⬅️ Sube tu archivo `.pth` en el panel izquierdo para comenzar el análisis.")
elif modelo and not archivos:
    st.info("📂 Ahora sube tus imágenes arriba para analizarlas.")
else:
    st.info("⬅️ Comienza subiendo tu modelo `.pth` en el panel izquierdo.")
