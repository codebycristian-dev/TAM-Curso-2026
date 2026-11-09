"""Tests del modelo: arquitectura, tabla y persistencia del artefacto.

La arquitectura esperada es la del cuaderno CodeTAM.ipynb (celda 6):
784 -> Dense 64 relu -> Dense 16 relu -> Dense 10 softmax.
"""

import numpy as np
import pytest

from tam_mnist.modelo import (
    MODEL_FILE_NAME,
    MODEL_VERSION,
    analyze_layer_structure,
    build_model,
    load_artifact,
    model_directory,
    save_artifact,
)


def test_architecture_matches_the_notebook():
    """Tres capas Dense con 64, 16 y 10 unidades y sus activaciones."""
    import tensorflow as tf

    model = build_model()
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    assert model.input_shape == (None, 784)
    assert model.output_shape == (None, 10)
    assert [int(layer.units) for layer in dense_layers] == [64, 16, 10]
    assert [layer.activation.__name__ for layer in dense_layers] == ["relu", "relu", "softmax"]
    assert model.count_params() == 784 * 64 + 64 + 64 * 16 + 16 + 16 * 10 + 10


def test_architecture_table_covers_input_and_dense_layers():
    """La tabla del cuaderno describe la entrada más las tres capas Dense."""
    table = analyze_layer_structure(build_model())

    assert [row["Capa"] for row in table] == ["entrada", "hidden_1", "hidden_2", "class_probabilities"]
    assert [row["Neuronas / unidades"] for row in table] == ["784 características", 64, 16, 10]
    assert sum(row["Parámetros"] for row in table[1:]) == build_model().count_params()


def test_artifact_round_trip_preserves_predictions(tmp_path):
    """Guardar y cargar devuelve un modelo que predice exactamente igual."""
    model = build_model()
    batch = np.linspace(0, 1, 784, dtype=np.float32).reshape(1, 784)
    expected = model.predict(batch, verbose=0)

    save_artifact(model, {"test_accuracy": 0.97}, tmp_path)
    loaded, metrics = load_artifact(tmp_path)

    assert (tmp_path / MODEL_FILE_NAME).is_file()
    assert metrics["test_accuracy"] == pytest.approx(0.97)
    assert metrics["model_version"] == MODEL_VERSION
    assert np.allclose(loaded.predict(batch, verbose=0), expected)


def test_loading_a_missing_artifact_returns_nothing(tmp_path):
    """Sin artefacto entrenado la app tiene que saberlo, sin excepción."""
    assert load_artifact(tmp_path) is None


def test_model_directory_is_configurable(monkeypatch, tmp_path):
    """El directorio del artefacto se puede redirigir, para tests y demos."""
    monkeypatch.setenv("TAM_MNIST_MODEL_DIR", str(tmp_path))
    assert model_directory() == tmp_path

    monkeypatch.delenv("TAM_MNIST_MODEL_DIR")
    assert model_directory().name == "modelo"
