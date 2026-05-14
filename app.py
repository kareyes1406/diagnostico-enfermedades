import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import zipfile

# ─────────────────────────────────────────────
#  Clases y colores
# ─────────────────────────────────────────────
CLASES_DEFAULT   = ['Black Spot', 'Fresh Leaf', 'Insectos', 'Mildew', 'Mosaico', 'Roya']
IMG_SIZE_DEFAULT = 300
MAX_IMAGENES     = 2000   # ← límite de imágenes por análisis

COLORES_CLASES = {
    'Black Spot': '#e74c3c',
    'Fresh Leaf': '#2ecc71',
    'Insectos':   '#f39c12',
    'Mildew':     '#9b59b6',
    'Mosaico':    '#3498db',
    'Roya':       '#e67e22',
}


# ─────────────────────────────────────────────
#  Modelo
# ─────────────────────────────────────────────
def cargar_modelo(ruta_pth):
    checkpoint = torch.load(ruta_pth, map_location="cpu")
    clases   = checkpoint.get("classes",  CLASES_DEFAULT)
    img_size = checkpoint.get("img_size", IMG_SIZE_DEFAULT)
    modelo   = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(clases))
    modelo.load_state_dict(checkpoint["model_state"])
    modelo.eval()
    return modelo, clases, img_size


def predecir(modelo, img_pil, clases, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(modelo(tensor), dim=1)[0]
        idx   = probs.argmax().item()
    return (
        clases[idx],
        round(float(probs[idx]) * 100, 1),
        {c: round(float(p) * 100, 1) for c, p in zip(clases, probs)},
    )


# ─────────────────────────────────────────────
#  Preprocesamiento: recortar imagen en tiles
# ─────────────────────────────────────────────
def recortar_en_tiles(img_pil, tile_size=300, overlap=0.2):
    """
    Divide una imagen grande (foto de dron) en recortes de tile_size x tile_size.
    overlap: porcentaje de solapamiento entre recortes (0.2 = 20%)
    """
    w, h  = img_pil.size
    paso  = int(tile_size * (1 - overlap))
    tiles = []

    for y in range(0, h - tile_size + 1, paso):
        for x in range(0, w - tile_size + 1, paso):
            recorte = img_pil.crop((x, y, x + tile_size, y + tile_size))
            tiles.append((recorte, x, y))

    # Si la imagen es más pequeña que tile_size, simplemente redimensionar
    if not tiles:
        tiles = [(img_pil.resize((tile_size, tile_size)), 0, 0)]

    return tiles


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico de Rosas 🌹",
    page_icon="🌹",
    layout="wide",
)

st.markdown("""
    <h1 style='text-align:center;'>🌹 Diagnóstico de Enfermedades en Rosas</h1>
    <p style='text-align:center; color:gray;'>
        EfficientNet-B3 · Precisión 99.88% · Compatible con fotos de dron · Hasta 2 000 imágenes
    </p><hr>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_file = st.file_uploader("Sube tu modelo (.pth)", type=["pth", "pt"])
    st.markdown("---")
    st.markdown("**Enfermedades detectadas:**")
    for c in CLASES_DEFAULT:
        color = COLORES_CLASES.get(c, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.85em;'>{c}</span> ",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Precisión del modelo: **99.88%**")

# Cargar modelo
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

# ── Pestañas ──────────────────────────────────
tab1, tab2 = st.tabs(["✂️ Paso 1 — Preprocesar fotos de dron", "🔬 Paso 2 — Diagnosticar"])


# ════════════════════════════════════════════
#  TAB 1 — PREPROCESAMIENTO
# ════════════════════════════════════════════
with tab1:
    st.subheader("✂️ Preprocesar fotos de dron → recortes 300×300")
    st.markdown(
        "Las fotos de dron son muy grandes y tomadas desde arriba. "
        "Esta herramienta las divide automáticamente en recortes de **300×300 px** "
        "listos para el modelo. Descárgalos como ZIP y súbelos en el Paso 2."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        overlap_pct = st.slider(
            "Solapamiento entre recortes",
            min_value=0, max_value=50, value=20, step=5,
            help="Un 20% significa que cada recorte comparte el 20% de área con el siguiente. "
                 "Más solapamiento = más recortes pero menos zonas perdidas en los bordes."
        )
    with col_cfg2:
        preview_max = st.number_input(
            "Recortes a previsualizar (máx)",
            min_value=4, max_value=40, value=12, step=4,
        )

    fotos_dron = st.file_uploader(
        "Sube las fotos del dron (JPG, PNG) — pueden ser varias a la vez",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="dron_uploader",
    )

    if fotos_dron:
        if st.button("✂️ Procesar y preparar recortes", type="primary", use_container_width=True):

            todos_los_tiles = []
            resumen         = []
            barra = st.progress(0, text="Procesando fotos del dron...")

            for i, foto in enumerate(fotos_dron):
                img   = Image.open(io.BytesIO(foto.read())).convert("RGB")
                w, h  = img.size
                tiles = recortar_en_tiles(img, tile_size=300, overlap=overlap_pct / 100)

                nombre_base = foto.name.rsplit(".", 1)[0]
                for idx_t, (tile, tx, ty) in enumerate(tiles):
                    nombre_tile = f"{nombre_base}_tile_{idx_t:04d}_x{tx}_y{ty}.jpg"
                    todos_los_tiles.append((nombre_tile, tile))

                resumen.append({
                    "Foto original":      foto.name,
                    "Tamaño original":    f"{w} × {h} px",
                    "Recortes generados": len(tiles),
                })

                barra.progress(
                    (i + 1) / len(fotos_dron),
                    text=f"Procesando {i+1}/{len(fotos_dron)}: {foto.name}",
                )

            barra.empty()

            st.success(f"✅ {len(todos_los_tiles)} recortes generados de {len(fotos_dron)} foto(s)")
            st.dataframe(pd.DataFrame(resumen), use_container_width=True)

            # Vista previa
            st.markdown(f"**Vista previa — primeros {min(preview_max, len(todos_los_tiles))} recortes:**")
            n_cols = 4
            muestra = todos_los_tiles[:preview_max]
            for fila in [muestra[i:i+n_cols] for i in range(0, len(muestra), n_cols)]:
                cols_prev = st.columns(n_cols)
                for j, (nombre, tile) in enumerate(fila):
                    with cols_prev[j]:
                        st.image(tile, use_container_width=True)
                        st.caption(nombre.split("_tile_")[1][:12])

            # ZIP para descargar
            st.markdown("---")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for nombre, tile in todos_los_tiles:
                    img_bytes = io.BytesIO()
                    tile.save(img_bytes, format="JPEG", quality=95)
                    zf.writestr(nombre, img_bytes.getvalue())
            zip_buffer.seek(0)

            st.download_button(
                label=f"⬇️ Descargar {len(todos_los_tiles)} recortes (.zip)",
                data=zip_buffer,
                file_name="recortes_dron_300x300.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.info("💡 Extrae el ZIP y sube las imágenes en la pestaña **Paso 2 — Diagnosticar**")


# ════════════════════════════════════════════
#  TAB 2 — DIAGNÓSTICO
# ════════════════════════════════════════════
with tab2:
    st.subheader("🔬 Diagnosticar imágenes")

    if not modelo_cargado:
        st.warning("⬅️ Primero sube tu modelo `.pth` en el panel izquierdo.")
    else:
        archivos = st.file_uploader(
            f"Sube hasta {MAX_IMAGENES} imágenes (JPG, PNG) — las del ZIP del Paso 1 o directamente tus fotos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="diag_uploader",
        )

        if len(archivos) > MAX_IMAGENES:
            st.warning(f"⚠️ Se recibieron {len(archivos)} imágenes — solo se procesarán las primeras {MAX_IMAGENES}.")
            archivos = archivos[:MAX_IMAGENES]

        if archivos:
            st.info(f"📂 {len(archivos)} imágenes listas para analizar.")

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
                        text=f"Analizando {i+1}/{len(archivos)}: {archivo.name}",
                    )

                barra.empty()
                df     = pd.DataFrame(resultados)
                conteo = Counter(df["Diagnóstico"])

                # Métricas
                st.markdown("---")
                st.subheader("📊 Resumen")
                cols = st.columns(len(clases_modelo) + 1)
                cols[0].metric("Total analizadas", len(df))
                for i, clase in enumerate(clases_modelo):
                    cols[i + 1].metric(clase, conteo.get(clase, 0))

                # Gráficas
                st.markdown("---")
                colores_lista = [COLORES_CLASES.get(c, "#888") for c in conteo.keys()]
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🍕 Distribución")
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

                # Galería — solo muestra las primeras 200 para no saturar el navegador
                st.markdown("---")
                st.subheader("🖼️ Galería con diagnóstico")
                MAX_GALERIA = 200
                if len(imagenes) > MAX_GALERIA:
                    st.info(f"ℹ️ Mostrando las primeras {MAX_GALERIA} imágenes en la galería. "
                            f"Los resultados completos están en la tabla y el CSV.")
                muestra_galeria = imagenes[:MAX_GALERIA]
                n_cols = 4
                for fila in [muestra_galeria[i:i+n_cols] for i in range(0, len(muestra_galeria), n_cols)]:
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
                    "⬇️ Descargar resultados CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="resultados_rosas.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("📂 Sube las imágenes arriba para comenzar el análisis.")
