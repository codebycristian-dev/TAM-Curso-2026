"""Tests de la app de Streamlit, ejecutados sin navegador con `AppTest`.

Comprueban el comportamiento visible: qué pasa cuando falta el artefacto y qué
muestra la app cuando el modelo ya está cargado.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tam_mnist.comparacion import (
    NOMBRE_CONVOLUCIONAL,
    NOMBRE_DENSA,
    build_conv_network,
    build_dense_network,
    save_network,
)
from tam_mnist.modelo import build_model, save_artifact

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 180


@pytest.fixture
def model_directory(tmp_path, monkeypatch):
    """Aísla el artefacto de cada test en un directorio temporal."""
    monkeypatch.setenv("TAM_MNIST_MODEL_DIR", str(tmp_path))
    return tmp_path


def test_app_without_artifact_asks_to_train_the_model(model_directory):
    """Sin artefacto la app no explota: explica cómo generarlo."""
    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert any("entrenar_modelo.py" in block.value for block in app.code)
    assert app.error, "la app tiene que avisar que falta el modelo"


def test_app_with_artifact_shows_the_six_tabs_and_the_metrics(model_directory):
    """Con artefacto la app muestra las tres entradas, las dos redes y la comparación."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1, "epochs": 5}, model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert len(app.tabs) == 6, "tres entradas, red densa, red convolucional y comparación"
    assert [tab.label for tab in app.tabs] == [
        "📤 Subir imagen",
        "✏️ Dibujar",
        "🎲 Muestra del test set",
        "🔵 Red densa",
        "🟣 Red convolucional",
        "⚖️ Comparación",
    ]
    metric_labels = [metric.label for metric in app.metric]
    assert "Exactitud en prueba" in metric_labels
    assert "Dígito predicho" in metric_labels

    # El modelo recién construido no está entrenado: la app predice igual y muestra
    # la etiqueta real del test set para comparar, sin inventar aciertos.
    assert app.dataframe, "la tabla de arquitectura del cuaderno"


def test_app_prediction_on_a_test_sample_is_a_valid_digit(model_directory):
    """Sobre una imagen del test set la app responde un dígito y su confianza."""
    save_artifact(build_model(), {"test_accuracy": 0.5, "test_loss": 2.0, "epochs": 1}, model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    prediction_metrics = [metric for metric in app.metric if metric.label == "Dígito predicho"]
    assert prediction_metrics, "la pestaña del test set predice una muestra al abrir"

    value = int(float(prediction_metrics[0].value))
    assert 0 <= value <= 9
    assert "%" in prediction_metrics[0].delta, "la confianza se muestra como porcentaje"

    chart = app.get("vega_lite_chart")
    assert chart, "el ranking de las diez probabilidades se muestra como gráfico"


# --- Las tres vistas nuevas (cuaderno scratchpad_3) -----------------------------


def _save_comparison_networks(directory):
    """Guarda las dos redes sin entrenar: alcanza para que las vistas se dibujen."""
    save_network(NOMBRE_DENSA, build_dense_network(), {"test_accuracy": 0.5}, directory)
    save_network(NOMBRE_CONVOLUCIONAL, build_conv_network(), {"test_accuracy": 0.5}, directory)


def test_comparison_views_without_artifacts_explain_how_to_train_them(model_directory):
    """Sin los artefactos nuevos las vistas avisan cómo generarlos, sin romper la app."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1}, model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert any("entrenar_comparacion.py" in block.value for block in app.code)
    assert app.error, "las vistas nuevas tienen que avisar que faltan las redes"


def test_comparison_tab_predicts_with_both_networks(model_directory):
    """La comparación muestra una predicción por red sobre la misma muestra del test set."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1}, model_directory)
    _save_comparison_networks(model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Red densa 64-64" in labels
    assert "Red convolucional" in labels

    predictions = [
        metric for metric in app.metric if metric.label in ("Red densa 64-64", "Red convolucional")
    ]
    for metric in predictions:
        assert 0 <= int(float(metric.value)) <= 9
        assert "%" in metric.delta


def test_convolutional_view_shows_filters_and_activation_maps(model_directory):
    """La vista convolucional suma sus filtros aprendidos y los mapas de activación."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1}, model_directory)
    _save_comparison_networks(model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    markdown = " ".join(block.value for block in app.markdown)
    assert "Filtros aprendidos" in markdown
    assert "Mapas de activación" in markdown

    # Las dos redes tienen su tabla de capas: la densa 64-64 y la convolucional de siete.
    tables = [frame.value for frame in app.dataframe]
    assert any("Convolucion_1" in str(frame) for frame in tables)
    assert any("densa_1" in str(frame) for frame in tables)
