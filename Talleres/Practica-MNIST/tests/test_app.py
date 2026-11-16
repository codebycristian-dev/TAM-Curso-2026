"""Tests de la app de Streamlit, ejecutados sin navegador con `AppTest`.

Comprueban el comportamiento visible: qué pasa cuando falta el artefacto, qué muestran las
cuatro pestañas cuando el modelo ya está cargado y qué predice cada red sobre una foto.
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
PHOTO = Path(__file__).resolve().parent.parent / "muestras-reales" / "2.jpeg"
TIMEOUT = 180


@pytest.fixture
def model_directory(tmp_path, monkeypatch):
    """Aísla el artefacto de cada test en un directorio temporal."""
    monkeypatch.setenv("TAM_MNIST_MODEL_DIR", str(tmp_path))
    return tmp_path


def _save_comparison_networks(directory):
    """Guarda las dos redes sin entrenar: alcanza para que las vistas se dibujen."""
    save_network(NOMBRE_DENSA, build_dense_network(), {"test_accuracy": 0.5}, directory)
    save_network(NOMBRE_CONVOLUCIONAL, build_conv_network(), {"test_accuracy": 0.5}, directory)


def _upload_photo_everywhere(app):
    """Carga la misma foto en la entrada y en las tres vistas."""
    for uploader in app.file_uploader:
        uploader.set_value(("2.jpeg", PHOTO.read_bytes(), "image/jpeg"))
    return app.run()


def test_app_without_artifact_asks_to_train_the_model(model_directory):
    """Sin artefacto la app no explota: explica cómo generarlo."""
    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert any("entrenar_modelo.py" in block.value for block in app.code)
    assert app.error, "la app tiene que avisar que falta el modelo"


def test_app_with_artifact_shows_the_four_tabs_and_the_metrics(model_directory):
    """La app queda con la entrada por foto y las tres vistas de las redes."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1, "epochs": 5}, model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "📤 Subir imagen",
        "🔵 Red densa",
        "🟣 Red convolucional",
        "⚖️ Comparación",
    ]
    metric_labels = [metric.label for metric in app.metric]
    assert "Exactitud en prueba" in metric_labels
    assert app.dataframe, "la tabla de arquitectura del cuaderno"


def test_app_asks_for_a_photo_before_predicting(model_directory):
    """Sin foto la app no inventa una predicción: la pide en cada vista."""
    save_artifact(build_model(), {"test_accuracy": 0.9, "test_loss": 0.2}, model_directory)
    _save_comparison_networks(model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert "Dígito predicho" not in [metric.label for metric in app.metric]
    assert any("Subí una foto" in info.value for info in app.info)


def test_comparison_views_without_artifacts_explain_how_to_train_them(model_directory):
    """Sin los artefactos nuevos las vistas avisan cómo generarlos, sin romper la app."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1}, model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert any("entrenar_comparacion.py" in block.value for block in app.code)
    assert app.error, "las vistas nuevas tienen que avisar que faltan las redes"


def test_uploaded_photo_is_predicted_by_both_networks(model_directory):
    """Con una foto subida, las dos redes predicen y la comparación muestra ambas."""
    save_artifact(build_model(), {"test_accuracy": 0.9, "test_loss": 0.2}, model_directory)
    _save_comparison_networks(model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()
    assert len(app.file_uploader) == 4, "la entrada y las tres vistas piden su foto"

    app = _upload_photo_everywhere(app)

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Red densa 64-64" in labels
    assert "Red convolucional" in labels

    for metric in app.metric:
        if metric.label in ("Red densa 64-64", "Red convolucional"):
            assert 0 <= int(float(metric.value)) <= 9, "la predicción es un dígito"
            assert "%" in metric.delta, "la confianza se muestra como porcentaje"


def test_convolutional_view_shows_filters_and_activation_maps(model_directory):
    """La vista convolucional suma sus filtros aprendidos y los mapas de activación."""
    save_artifact(build_model(), {"test_accuracy": 0.9, "test_loss": 0.2}, model_directory)
    _save_comparison_networks(model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()
    app = _upload_photo_everywhere(app)

    assert not app.exception
    markdown = " ".join(block.value for block in app.markdown)
    assert "Filtros aprendidos" in markdown
    assert "Mapas de activación" in markdown

    # Las dos redes tienen su tabla de capas: la densa 64-64 y la convolucional de siete.
    tables = [str(frame.value) for frame in app.dataframe]
    assert any("Convolucion_1" in table for table in tables)
    assert any("densa_1" in table for table in tables)
