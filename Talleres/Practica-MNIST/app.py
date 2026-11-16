"""App de Streamlit de la práctica MNIST: subir, dibujar o elegir un dígito.

Además de la red densa 64-16 del cuaderno `CodeTAM.ipynb`, la app muestra las dos redes
del cuaderno de clase `experimentos/scratchpad_3.ipynb`: la densa 64-64 y la
convolucional, cada una con su predicción y su anatomía, y una vista que las compara.

Ejecutar desde la carpeta `Practica-MNIST/` con el entorno que tiene TensorFlow:

    streamlit run app.py

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
from tam_mnist.preproceso import (
    MENSAJE_SIN_DIGITO,
    Prediction,
    predict_probabilities,
    preprocess_bytes,
    preprocess_canvas,
)

PAGE_TITLE = "Práctica MNIST — TAM 2026"
CANVAS_PIXELS = 280
MENSAJE_SIN_MODELO = (
    "Todavía no hay un modelo entrenado. Generá el artefacto una vez y volvé a cargar esta página."
)
COMANDO_ENTRENAMIENTO = "python entrenar_modelo.py"
COMANDO_COMPARACION = "python entrenar_comparacion.py"
MENSAJE_SIN_REDES = (
    "Todavía no están entrenadas las dos redes de comparación. Generalas una vez y volvé a "
    "cargar esta página."
)
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp", "gif"]


st.set_page_config(page_title=PAGE_TITLE, page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo entrenado…")
def load_trained_model(directory: str):
    """Carga el artefacto una sola vez por sesión."""
    return load_artifact(Path(directory))


@st.cache_data(show_spinner="Cargando el conjunto de prueba de MNIST…")
def load_test_set():
    """Trae las 10 000 imágenes de prueba ya normalizadas, desde el caché local."""
    from tam_mnist.modelo import load_dataset

    _, (x_test, y_test) = load_dataset()
    return x_test, y_test


@st.cache_resource(show_spinner="Cargando las dos redes de comparación…")
def load_comparison_networks(directory: str):
    """Carga los dos artefactos de comparación una sola vez por sesión."""
    return load_networks(Path(directory))


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


def render_upload_tab(model) -> None:
    """Pestaña 1: subir una o varias imágenes PNG o JPG."""
    st.subheader("Subí una imagen de un dígito escrito a mano")
    st.write(
        "Sirve una foto o un recorte del dígito: la app lo pasa a escala de grises, invierte el "
        "fondo si hace falta, recorta el trazo y lo centra en 28 × 28 píxeles, igual que el cuaderno."
    )

    uploaded_files = st.file_uploader(
        "Imágenes del dígito",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        help="Podés subir varias imágenes a la vez; cada una se predice por separado.",
    )

    if not uploaded_files:
        st.info("Todavía no subiste ninguna imagen.")
        return

    for uploaded_file in uploaded_files:
        st.divider()
        st.markdown(f"**Archivo:** `{uploaded_file.name}`")
        try:
            vector, preview = preprocess_bytes(uploaded_file.getvalue())
        except ValueError as error:
            st.error(str(error))
            continue

        render_prediction(preview, predict_probabilities(model, vector), uploaded_file.name)


def render_canvas_tab(model) -> None:
    """Pestaña 2: dibujar el dígito con el mouse."""
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.warning(
            "Falta el componente de dibujo. Instalalo con "
            "`python -m pip install streamlit-drawable-canvas` o usá las otras pestañas."
        )
        return

    st.subheader("Dibujá un dígito con el mouse")
    st.write("Trazá el número en el lienzo negro. La predicción se actualiza al soltar el trazo.")

    if "canvas_generation" not in st.session_state:
        st.session_state.canvas_generation = 0

    canvas_column, result_column = st.columns([1, 2], gap="large")
    with canvas_column:
        canvas = st_canvas(
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="rgba(0, 0, 0, 0)",
            height=CANVAS_PIXELS,
            width=CANVAS_PIXELS,
            drawing_mode="freedraw",
            update_streamlit=True,
            return_image_data=True,
            key=f"digit_canvas_{st.session_state.canvas_generation}",
        )
        if st.button("Limpiar lienzo", width="stretch"):
            st.session_state.canvas_generation += 1
            st.rerun()

    with result_column:
        if canvas is None or canvas.image_data is None:
            st.info("Dibujá un dígito para ver la predicción.")
            return

        try:
            vector, preview = preprocess_canvas(canvas.image_data)
        except ValueError as error:
            message = str(error)
            if message == MENSAJE_SIN_DIGITO:
                st.info("El lienzo está vacío: dibujá un dígito.")
            else:
                st.error(message)
            return

        render_prediction(preview, predict_probabilities(model, vector), "Tu dibujo, normalizado")


def render_test_set_tab(model) -> None:
    """Pestaña 3: probar con una imagen del conjunto de prueba, que el modelo no entrenó."""
    st.subheader("Probá con una imagen del conjunto de prueba")
    st.write(
        "Estas imágenes no participaron del entrenamiento. Comparar la etiqueta real con la "
        "predicción es la forma más honesta de ver cómo anda el modelo."
    )

    x_test, y_test = load_test_set()

    control_column, random_column = st.columns([3, 1])
    with control_column:
        index = st.slider(
            "Índice de la imagen de prueba",
            min_value=0,
            max_value=int(x_test.shape[0]) - 1,
            value=0,
            step=1,
        )
    with random_column:
        st.write("")
        st.write("")
        if st.button("Elegir al azar", width="stretch"):
            st.session_state.test_index = int(pd.Series(range(int(x_test.shape[0]))).sample(1).iloc[0])
            st.rerun()

    if "test_index" in st.session_state:
        index = st.session_state.test_index

    preview = x_test[index].reshape(28, 28)
    actual_digit = int(y_test[index])
    prediction = predict_probabilities(model, x_test[index : index + 1])

    render_prediction(preview, prediction, f"Etiqueta real: {actual_digit} (índice {index})")
    if prediction.digit == actual_digit:
        st.success(f"Correcto: el modelo acertó el {prediction.confidence:.1%} con la etiqueta real.")
    else:
        st.error(
            f"Error: la etiqueta real es {actual_digit} y el modelo respondió {prediction.digit} "
            f"con {prediction.confidence:.1%} de confianza."
        )


@dataclass(frozen=True)
class NetworkInput:
    """La entrada elegida para una vista: los píxeles, la vista previa y la etiqueta real."""

    pixels: np.ndarray
    preview: np.ndarray
    caption: str
    actual_digit: int | None


def render_network_input(prefix: str) -> NetworkInput | None:
    """Selector de entrada compartido por las tres vistas de comparación."""
    source = st.radio(
        "Entrada",
        ["Muestra del test set", "Subir una foto"],
        key=f"{prefix}_source",
        horizontal=True,
    )

    if source == "Subir una foto":
        uploaded = st.file_uploader(
            "Imagen del dígito",
            type=IMAGE_TYPES,
            key=f"{prefix}_file",
            help="Una imagen por vez en estas vistas, para comparar las redes sobre el mismo dígito.",
        )
        if uploaded is None:
            st.info("Subí una foto del dígito para ver la predicción y sus activaciones.")
            return None
        try:
            pixels, preview = preprocess_bytes(uploaded.getvalue())
        except ValueError as error:
            st.error(str(error))
            return None
        return NetworkInput(
            pixels=pixels, preview=preview, caption=uploaded.name, actual_digit=None
        )

    x_test, y_test = load_test_set()
    index = st.slider(
        "Índice de la imagen de prueba",
        min_value=0,
        max_value=int(x_test.shape[0]) - 1,
        value=0,
        step=1,
        key=f"{prefix}_index",
    )
    actual_digit = int(y_test[index])
    return NetworkInput(
        pixels=x_test[index : index + 1],
        preview=x_test[index].reshape(28, 28),
        caption=f"Etiqueta real: {actual_digit} (índice {index})",
        actual_digit=actual_digit,
    )


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


def render_verdict(prediction: Prediction, actual_digit: int | None) -> None:
    """Dice si la predicción coincide con la etiqueta real, cuando la hay."""
    if actual_digit is None:
        return
    if prediction.digit == actual_digit:
        st.success(f"Acertó: la etiqueta real es {actual_digit}.")
    else:
        st.error(f"Falló: la etiqueta real es {actual_digit} y respondió {prediction.digit}.")


def render_network_view(name: str, artifact) -> None:
    """Vista de una red: la predicción arriba y la anatomía abajo."""
    st.subheader(REDES[name].titulo)

    if artifact is None:
        st.error(MENSAJE_SIN_REDES)
        st.code(COMANDO_COMPARACION, language="bash")
        return

    model, metrics = artifact
    st.caption(describe_metrics(metrics))

    data = render_network_input(name)
    if data is not None:
        prediction = predict_with(name, model, data.pixels)
        render_prediction(data.preview, prediction, data.caption)
        render_verdict(prediction, data.actual_digit)

    st.divider()
    render_anatomy(name, model, data)


def render_comparison_tab(artifacts: dict) -> None:
    """Vista de comparación: las dos predicciones sobre la misma entrada y el contraste."""
    st.subheader("Las dos redes sobre la misma entrada")
    st.write(
        "La imagen se prepara una sola vez. La densa recibe 784 píxeles sueltos; la "
        "convolucional, la imagen de 28 × 28 con su vecindad."
    )

    if any(artifact is None for artifact in artifacts.values()):
        st.error(MENSAJE_SIN_REDES)
        st.code(COMANDO_COMPARACION, language="bash")
        return

    data = render_network_input("comparacion")
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

        if data.actual_digit is not None:
            hits = [
                REDES[name].titulo
                for name, prediction in predictions.items()
                if prediction.digit == data.actual_digit
            ]
            if len(hits) == len(predictions):
                st.caption("Las dos aciertan la etiqueta real.")
            elif hits:
                st.caption(f"Sólo acierta {hits[0].lower()}.")
            else:
                st.caption("Ninguna de las dos acierta la etiqueta real.")

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
        "La red densa del cuaderno `CodeTAM.ipynb` predice qué dígito escribiste. "
        "Los tres caminos de abajo terminan en la misma predicción. Las otras tres pestañas "
        "muestran las redes del cuaderno de clase `experimentos/scratchpad_3.ipynb`: la densa, "
        "la convolucional y la comparación de las dos."
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

    upload_tab, canvas_tab, test_set_tab, dense_tab, conv_tab, comparison_tab = st.tabs(
        [
            "📤 Subir imagen",
            "✏️ Dibujar",
            "🎲 Muestra del test set",
            "🔵 Red densa",
            "🟣 Red convolucional",
            "⚖️ Comparación",
        ]
    )
    with upload_tab:
        render_upload_tab(model)
    with canvas_tab:
        render_canvas_tab(model)
    with test_set_tab:
        render_test_set_tab(model)
    with dense_tab:
        render_network_view(NOMBRE_DENSA, comparison_networks[NOMBRE_DENSA])
    with conv_tab:
        render_network_view(NOMBRE_CONVOLUCIONAL, comparison_networks[NOMBRE_CONVOLUCIONAL])
    with comparison_tab:
        render_comparison_tab(comparison_networks)


main()
