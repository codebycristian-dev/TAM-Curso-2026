"""App de Streamlit de la práctica MNIST: subir, dibujar o elegir un dígito.

Ejecutar desde la carpeta `Practica-MNIST/` con el entorno que tiene TensorFlow:

    streamlit run app.py

La app sólo carga el artefacto entrenado (`modelo/mnist_digits.keras`). Para
generarlo o regenerarlo se ejecuta `python entrenar_modelo.py`.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

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
        type=["png", "jpg", "jpeg", "bmp", "webp", "gif"],
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
        "Los tres caminos de abajo terminan en la misma predicción."
    )

    artifact = load_trained_model(str(model_directory()))
    if artifact is None:
        st.error(MENSAJE_SIN_MODELO)
        st.code(COMANDO_ENTRENAMIENTO, language="bash")
        st.caption(f"Busqué el artefacto en `{model_directory()}`.")
        st.stop()

    model, metrics = artifact
    render_sidebar(model, metrics)

    upload_tab, canvas_tab, test_set_tab = st.tabs(
        ["📤 Subir imagen", "✏️ Dibujar", "🎲 Muestra del test set"]
    )
    with upload_tab:
        render_upload_tab(model)
    with canvas_tab:
        render_canvas_tab(model)
    with test_set_tab:
        render_test_set_tab(model)


main()
