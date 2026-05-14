import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import plotly.express as px
from collections import Counter
import io

# ─────────────────────────────────────────────
#  Clases y colores del modelo de rosas
# ─────────────────────────────────────────────
CLASES_DEFAULT = ['Black Spot', 'Fresh Leaf', 'Insectos', 'Mildew', 'Mosaico', 'Roya']
IMG_SIZE_DEFAULT = 300

COLORES_CLASES = {
    'Black Spot':  '#e74c3c',
    'Fresh Leaf':  '#2ecc71',
    'Insectos':    '#f39c12',
    'Mildew':      '#9b59b6',
    'Mosaico':     '#3498db',
    'Roya':        '#e67e22',
}


def cargar_modelo(ruta_pth):
    checkpoint = torch.load(ruta_pth, map_location="cpu")
    clases   = checkpoint.get("classes",  CLASES_DEFAULT)
    img_size = checkpoint.get("img_size", IMG_SIZE_DEFAULT)

    modelo = models.efficientnet_b3(weights=None)
    in_features = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(clases)),
    )
    modelo.load_state_dict(checkpoint["model_state"])
    modelo.eval()
    return modelo, clases, img_size


def predecir(modelo, img_pil, clases, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        salida = modelo(tensor)
        probs  = torch.softmax(salida, dim=1)[0]
        idx    = probs.argmax().item()
    return (
        clases[idx],
        round(float(probs[idx]) * 100, 1),
        {c: round(float(p) * 100, 1) for c, p in zip(clases, probs)},
    )


# ── Página ────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico de Rosas 🌹",
    page_icon="🌹",
    layout="wide",
)

st.markdown("""
    <h1 style='text-align:center;'>🌹 Diagnóstico de Enfermedades en Rosas</h1>
    <p style='text-align:center; color:gray;'>
        Sube hasta 500 imágenes — modelo EfficientNet-B3 · Precisión 99.88%
    </p><hr>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_file = st.file_uploader("1. Sube tu modelo (.pth)", type=["pth", "pt"])
    st.markdown("---")
    st.markdown("**Enfermedades detectadas:**")
    for c in CLASES_DEFAULT:
        color = COLORES_CLASES.get(c, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.85em;'>{c}</span> ",
            unsafe_allow_html=True,
        )

# ── Carga del modelo ──────────────────────────
modelo_cargado  = None
clases_modelo   = CLASES_DEFAULT
img_size_modelo = IMG_SIZE_DEFAULT

if modelo_file:
    with st.spinner("Cargando modelo..."):
        ruta_tmp = "/tmp/modelo_rosas.pth"
        with open(ruta_tmp, "wb") as f:
            f.write(modelo_file.read())
        try:
            modelo_cargado, clases_modelo, img_size_modelo = cargar_modelo(ruta_tmp)
            st.sidebar.success(f"✅ Modelo listo — {len(clases_modelo)} clases · {img_size_modelo}px")
        except Exception as e:
            st.sidebar.error(f"❌ Error:\n{e}")

# ── Subida de imágenes ────────────────────────
st.subheader("📁 Sube tus imágenes")
archivos = st.file_uploader(
    "Selecciona hasta 500 imágenes (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
if len(archivos) > 500:
    st.warning("⚠️ Solo se procesarán las primeras 500 imágenes.")
    archivos = archivos[:500]

# ── Botón de análisis ─────────────────────────
if archivos and modelo_cargado:
    if st.button("🚀 Analizar imágenes", type="primary", use_container_width=True):

        resultados = []
        imagenes   = []
        barra      = st.progress(0, text="Analizando...")

        for i, archivo in enumerate(archivos):
            img = Image.open(io.BytesIO(archivo.read()))
            clase, confianza, probs_dict = predecir(
                modelo_cargado, img, clases_modelo, img_size_modelo
            )
            resultados.append({
                "Imagen":        archivo.name,
                "Diagnóstico":   clase,
                "Confianza (%)": confianza,
                **{f"P({c}) %": v for c, v in probs_dict.items()},
            })
            imagenes.append((archivo.name, img, clase, confianza))
            barra.progress(
                (i + 1) / len(archivos),
                text=f"Procesando {i+1}/{len(archivos)}: {archivo.name}",
            )

        barra.empty()
        df     = pd.DataFrame(resultados)
        conteo = Counter(df["Diagnóstico"])

        # Métricas
        st.markdown("---")
        st.subheader("📊 Resumen")
        cols = st.columns(len(clases_modelo) + 1)
        cols[0].metric("Total", len(df))
        for i, clase in enumerate(clases_modelo):
            cols[i + 1].metric(clase, conteo.get(clase, 0))

        # Gráficas
        st.markdown("---")
        colores_lista = [COLORES_CLASES.get(c, "#888") for c in conteo.keys()]
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🍕 Distribución por diagnóstico")
            fig_pie = px.pie(
                values=list(conteo.values()),
                names=list(conteo.keys()),
                color_discrete_sequence=colores_lista,
                hole=0.35,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("📊 Cantidad por diagnóstico")
            fig_bar = px.bar(
                x=list(conteo.keys()),
                y=list(conteo.values()),
                color=list(conteo.keys()),
                color_discrete_sequence=colores_lista,
                labels={"x": "Diagnóstico", "y": "Cantidad"},
                text_auto=True,
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Galería
        st.markdown("---")
        st.subheader("🖼️ Galería con diagnóstico")
        n_cols = 4
        for fila in [imagenes[i:i+n_cols] for i in range(0, len(imagenes), n_cols)]:
            cols_gal = st.columns(n_cols)
            for j, (nombre, img, clase, conf) in enumerate(fila):
                with cols_gal[j]:
                    st.image(img, use_container_width=True)
                    color = COLORES_CLASES.get(clase, "#888")
                    st.markdown(
                        f"<div style='text-align:center;margin-top:4px;'>"
                        f"<span style='background:{color};color:white;padding:2px 10px;"
                        f"border-radius:12px;font-size:0.8em;'>{clase}</span><br>"
                        f"<small style='color:gray;'>{conf}% confianza</small></div>",
                        unsafe_allow_html=True,
                    )

        # Tabla y descarga
        st.markdown("---")
        st.subheader("📋 Tabla completa de resultados")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="resultados_rosas.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif archivos and not modelo_cargado:
    st.info("⬅️ Sube tu modelo `.pth` en el panel izquierdo para continuar.")
elif modelo_cargado and not archivos:
    st.info("📂 Modelo listo. Sube tus imágenes arriba para analizarlas.")
else:
    st.info("⬅️ Empieza subiendo tu modelo `.pth` en el panel izquierdo.")
