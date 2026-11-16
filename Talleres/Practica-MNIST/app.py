"""App de Streamlit de la práctica MNIST: subir una foto de un dígito.

Además de la red densa 64-16 del cuaderno `CodeTAM.ipynb`, la app muestra las dos redes
del cuaderno de clase `experimentos/scratchpad_3.ipynb`: la densa 64-64 y la
convolucional, cada una con su predicción y su anatomía, y una vista que las compara.

Ejecutar desde la carpeta `Practica-MNIST/` con el entorno que tiene TensorFlow:

    streamlit run app.py

La foto se sube una sola vez, en una sección de carga por encima de las pestañas: la
pestaña `Subir imagen` la predice con la red densa 64-16 y las tres vistas de las otras
redes leen esa misma foto, sin pedir una propia.

La app sólo carga artefactos entrenados: `modelo/mnist_digits.keras` para la red
densa 64-16 y `modelo/comparacion/` para las dos redes comparadas. Se generan con
`python entrenar_modelo.py` y `python entrenar_comparacion.py`.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tam_mnist.comparacion import (
    NOMBRE_CONVOLUCIONAL,
    NOMBRE_DENSA,
    REDES,
    activation_maps,
    adapt_batch,
    first_conv_filters,
    layer_table,
    load_networks,
)
from tam_mnist.modelo import analyze_layer_structure, load_artifact, model_directory
from tam_mnist.preproceso import Prediction, predict_probabilities, preprocess_bytes

PAGE_TITLE = "Práctica MNIST — TAM 2026"
MENSAJE_SIN_MODELO = (
    "Todavía no hay un modelo entrenado. Generá el artefacto una vez y volvé a cargar esta página."
)
COMANDO_ENTRENAMIENTO = "python entrenar_modelo.py"
COMANDO_COMPARACION = "python entrenar_comparacion.py"
MENSAJE_SIN_REDES = (
    "Todavía no están entrenadas las dos redes de comparación. Generalas una vez y volvé a "
    "cargar esta página."
)
MENSAJE_SIN_FOTO = "Subí una foto del dígito para ver la predicción y sus activaciones."
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp", "gif"]


st.set_page_config(page_title=PAGE_TITLE, page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo entrenado…")
def load_trained_model(directory: str):
    """Carga el artefacto una sola vez por sesión."""
    return load_artifact(Path(directory))


@st.cache_resource(show_spinner="Cargando las dos redes de comparación…")
def load_comparison_networks(directory: str):
    """Carga los dos artefactos de comparación una sola vez por sesión."""
    return load_networks(Path(directory))


@st.cache_data(show_spinner=False)
def preprocess_photo(name: str, payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesa una foto una sola vez por nombre y contenido.

    La clave es el nombre y los bytes: la misma imagen se prepara una vez por corrida y la
    comparten la pestaña de subir y las tres vistas.
    """
    return preprocess_bytes(payload)


def render_probabilities(prediction: Prediction) -> None:
    """Muestra el ranking de probabilidades de las diez clases."""
    frame = pd.DataFrame(
        {"probabilidad": [probability for _, probability in prediction.ranking]},
        index=[digit for digit, _ in prediction.ranking],
    )
    st.bar_chart(frame, height=240, y_label="probabilidad")


def render_prediction(preview, prediction: Prediction, caption: str) -> None:
    """Presenta la imagen preprocesada junto al dígito predicho y sus probabilidades."""
    image_column, result_column = st.columns([1, 3], gap="large")

    with image_column:
        st.image(preview, caption=caption, width=200, clamp=True)

    with result_column:
        st.metric(
            label="Dígito predicho",
            value=prediction.digit,
            delta=f"{prediction.confidence:.1%} de confianza",
            delta_color="off",
        )
        runner_up, runner_up_probability = prediction.ranking[1]
        st.caption(
            f"Segunda opción: {runner_up} con {runner_up_probability:.1%}. "
            "El ranking completo usa la misma salida softmax de diez clases del cuaderno."
        )
        render_probabilities(prediction)


def render_upload_tab(model, uploaded_files: list) -> None:
    """Pestaña 1: predice con la red densa 64-16 cada una de las imágenes subidas."""
    st.subheader("Subí una imagen de un dígito escrito a mano")
    st.write(
        "Sirve una foto o un recorte del dígito: la app lo pasa a escala de grises, invierte el "
        "fondo si hace falta, recorta el trazo y lo centra en 28 × 28 píxeles, igual que el cuaderno."
    )

    if not uploaded_files:
        st.info("Todavía no subiste ninguna imagen.")
        return

    for uploaded_file in uploaded_files:
        st.divider()
        st.markdown(f"**Archivo:** `{uploaded_file.name}`")
        try:
            vector, preview = preprocess_photo(uploaded_file.name, uploaded_file.getvalue())
        except ValueError as error:
            st.error(str(error))
            continue

        render_prediction(preview, predict_probabilities(model, vector), uploaded_file.name)


@dataclass(frozen=True)
class NetworkInput:
    """La foto ya preprocesada: los píxeles que recibe la red y la vista de 28 × 28."""

    pixels: np.ndarray
    preview: np.ndarray
    caption: str


def render_shared_input() -> tuple[list, NetworkInput | None]:
    """Única carga de la página: devuelve los archivos subidos y la foto que leen las vistas.

    Va por encima de las pestañas para que su valor esté disponible en todas en la misma
    corrida. Con varias imágenes se elige una sola vez cuál analizan las tres vistas.
    """
    uploaded_files = st.file_uploader(
        "Imágenes del dígito",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        help="Podés subir varias imágenes; las vistas analizan la que elijas acá.",
    )
    if not uploaded_files:
        st.info(MENSAJE_SIN_FOTO)
        return [], None

    names = [uploaded_file.name for uploaded_file in uploaded_files]
    chosen = uploaded_files[0]
    if len(names) > 1:
        chosen_name = st.selectbox(
            "Foto que analizan las tres vistas",
            names,
            help="La pestaña de subir predice todas; las vistas analizan sólo esta.",
        )
        chosen = uploaded_files[names.index(chosen_name)]

    try:
        pixels, preview = preprocess_photo(chosen.name, chosen.getvalue())
    except ValueError as error:
        st.error(str(error))
        return uploaded_files, None

    return uploaded_files, NetworkInput(pixels=pixels, preview=preview, caption=chosen.name)


def predict_with(name: str, model, pixels: np.ndarray) -> Prediction:
    """Predice con una de las dos redes, llevando los píxeles a la forma que espera."""
    return predict_probabilities(model, adapt_batch(name, pixels))


def describe_metrics(metrics: dict) -> str:
    """Una línea con cómo se entrenó la red y cuánto acierta en el conjunto de prueba."""
    if not metrics:
        return "Esta red no tiene métricas guardadas."
    accuracy = metrics.get("test_accuracy")
    accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "?"
    return (
        f"Entrenada con {metrics.get('epochs', '?')} épocas y lotes de "
        f"{metrics.get('batch_size', '?')}; exactitud en prueba {accuracy_text}."
    )


def render_matrix_grid(matrices: np.ndarray, columns: int, cell: float = 0.9) -> None:
    """Dibuja una grilla de matrices en escala de grises, sin ejes."""
    count = len(matrices)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * cell, rows * cell), squeeze=False)

    for axis, matrix in zip(axes.ravel(), matrices):
        axis.imshow(matrix, cmap="gray", interpolation="nearest")
        axis.axis("off")
    for axis in axes.ravel()[count:]:
        axis.axis("off")

    figure.tight_layout(pad=0.15)
    st.pyplot(figure)
    plt.close(figure)


def render_anatomy(name: str, model, data: NetworkInput | None) -> None:
    """Anatomía de la red: capas y parámetros; en la convolucional, filtros y activaciones."""
    st.dataframe(pd.DataFrame(layer_table(model)).astype(str), hide_index=True, width="stretch")
    st.caption(f"Parámetros entrenables: {model.count_params():,}")

    if name != NOMBRE_CONVOLUCIONAL:
        st.caption(
            "La red densa no tiene estructura espacial: cada píxel entra como una característica "
            "suelta, así que no hay filtros ni mapas de activación que mostrar."
        )
        return

    st.markdown("**Filtros aprendidos por `Convolucion_1`**")
    st.caption(
        "Los 32 filtros de 3 × 3: el patrón que cada uno busca en cada parche de la imagen."
    )
    render_matrix_grid(first_conv_filters(model), columns=8, cell=0.7)

    st.markdown("**Mapas de activación de `Convolucion_2`**")
    if data is None:
        st.info("Elegí una entrada para ver cómo responde cada filtro a ese dígito.")
        return
    st.caption(
        "Los 64 mapas que deja la segunda convolución para esa entrada: dónde se enciende "
        "cada filtro."
    )
    render_matrix_grid(activation_maps(model, data.pixels), columns=8, cell=0.7)


def render_network_view(name: str, artifact, data: NetworkInput | None) -> None:
    """Vista de una red: la predicción arriba y la anatomía abajo.

    La foto llega ya preprocesada desde la carga compartida: la vista no pide ninguna.
    """
    st.subheader(REDES[name].titulo)

    if artifact is None:
        st.error(MENSAJE_SIN_REDES)
        st.code(COMANDO_COMPARACION, language="bash")
        return

    model, metrics = artifact
    st.caption(describe_metrics(metrics))

    if data is not None:
        prediction = predict_with(name, model, data.pixels)
        render_prediction(data.preview, prediction, data.caption)

    st.divider()
    render_anatomy(name, model, data)


def render_comparison_tab(artifacts: dict, data: NetworkInput | None) -> None:
    """Vista de comparación: las dos predicciones sobre la misma entrada y el contraste.

    La entrada llega ya preprocesada desde la carga compartida de la página.
    """
    st.subheader("Las dos redes sobre la misma entrada")
    st.write(
        "La imagen se prepara una sola vez. La densa recibe 784 píxeles sueltos; la "
        "convolucional, la imagen de 28 × 28 con su vecindad."
    )

    if any(artifact is None for artifact in artifacts.values()):
        st.error(MENSAJE_SIN_REDES)
        st.code(COMANDO_COMPARACION, language="bash")
        return

    if data is not None:
        predictions = {
            name: predict_with(name, artifacts[name][0], data.pixels) for name in artifacts
        }

        preview_column, dense_column, conv_column = st.columns([1, 2, 2], gap="large")
        with preview_column:
            st.image(data.preview, caption=data.caption, width=140, clamp=True)
        for column, name in ((dense_column, NOMBRE_DENSA), (conv_column, NOMBRE_CONVOLUCIONAL)):
            prediction = predictions[name]
            with column:
                st.metric(
                    label=REDES[name].titulo,
                    value=prediction.digit,
                    delta=f"{prediction.confidence:.1%} de confianza",
                    delta_color="off",
                )

        dense = predictions[NOMBRE_DENSA]
        conv = predictions[NOMBRE_CONVOLUCIONAL]
        if dense.digit == conv.digit:
            st.success(f"Las dos redes coinciden en el dígito {dense.digit}.")
        else:
            st.warning(
                f"Las dos redes no coinciden: la densa dice {dense.digit} y la convolucional "
                f"{conv.digit}."
            )

    st.divider()
    st.markdown("**Arquitectura y resultados**")
    rows = []
    for name, (model, metrics) in artifacts.items():
        accuracy = metrics.get("test_accuracy")
        rows.append(
            {
                "Red": REDES[name].titulo,
                "Entrada": " × ".join(str(size) for size in REDES[name].forma_entrada),
                "Parámetros": f"{model.count_params():,}",
                "Exactitud en prueba": f"{accuracy:.2%}" if accuracy is not None else "?",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "La convolucional tiene menos parámetros porque cada pooling achica la imagen antes de "
        "aplanarla: no necesita una conexión por píxel."
    )


def _format_count(value) -> str:
    """Formatea un conteo con separador de miles, o `?` si la métrica no existe."""
    return f"{value:,}" if isinstance(value, int) else "?"


def render_sidebar(model, metrics: dict) -> None:
    """Métricas del artefacto y la tabla de arquitectura del cuaderno."""
    with st.sidebar:
        st.header("El modelo")
        accuracy = metrics.get("test_accuracy")
        if accuracy is not None:
            st.metric("Exactitud en prueba", f"{accuracy:.2%}")
        loss = metrics.get("test_loss")
        if loss is not None:
            st.metric("Pérdida en prueba", f"{loss:.4f}")

        st.caption(
            f"Entrenado con {metrics.get('epochs', '?')} épocas, lotes de "
            f"{metrics.get('batch_size', '?')} y {_format_count(metrics.get('test_size'))} imágenes de prueba."
        )

        table = analyze_layer_structure(model)
        st.subheader("Arquitectura")
        # Arrow no admite una columna con texto y números mezclados, así que la tabla
        # se muestra como texto: la fila de entrada dice "784 características".
        st.dataframe(
            pd.DataFrame(table).astype(str), hide_index=True, width="stretch"
        )
        st.caption(f"Parámetros entrenables: {model.count_params():,}")

        st.divider()
        st.caption(f"Artefacto: `{model_directory()}`")
        if st.button("Volver a cargar el modelo", width="stretch"):
            load_trained_model.clear()
            st.rerun()


def main() -> None:
    st.title("🔢 Práctica MNIST: dígitos escritos a mano")
    st.write(
        "La red densa del cuaderno `CodeTAM.ipynb` predice el dígito de la foto que subas. "
        "Las otras tres pestañas muestran las redes del cuaderno de clase "
        "`experimentos/scratchpad_3.ipynb`: la densa, la convolucional y la comparación de las dos."
    )

    artifact = load_trained_model(str(model_directory()))
    if artifact is None:
        st.error(MENSAJE_SIN_MODELO)
        st.code(COMANDO_ENTRENAMIENTO, language="bash")
        st.caption(f"Busqué el artefacto en `{model_directory()}`.")
        st.stop()

    model, metrics = artifact
    render_sidebar(model, metrics)
    comparison_networks = load_comparison_networks(str(model_directory()))

    uploaded_files, data = render_shared_input()

    upload_tab, dense_tab, conv_tab, comparison_tab = st.tabs(
        ["📤 Subir imagen", "🔵 Red densa", "🟣 Red convolucional", "⚖️ Comparación"]
    )
    with upload_tab:
        render_upload_tab(model, uploaded_files)
    with dense_tab:
        render_network_view(NOMBRE_DENSA, comparison_networks[NOMBRE_DENSA], data)
    with conv_tab:
        render_network_view(NOMBRE_CONVOLUCIONAL, comparison_networks[NOMBRE_CONVOLUCIONAL], data)
    with comparison_tab:
        render_comparison_tab(comparison_networks, data)


main()
