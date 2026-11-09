"""Tests de la app de Streamlit, ejecutados sin navegador con `AppTest`.

Comprueban el comportamiento visible: qué pasa cuando falta el artefacto y qué
muestra la app cuando el modelo ya está cargado.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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


def test_app_with_artifact_shows_the_three_inputs_and_the_metrics(model_directory):
    """Con artefacto la app muestra las tres entradas y las métricas del cuaderno."""
    save_artifact(build_model(), {"test_accuracy": 0.97, "test_loss": 0.1, "epochs": 5}, model_directory)

    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()

    assert not app.exception
    assert len(app.tabs) == 3, "subir imagen, dibujar y muestra del test set"
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
