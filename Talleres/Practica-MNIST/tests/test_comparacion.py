"""Tests de las dos redes del cuaderno `experimentos/scratchpad_3.ipynb`.

La densa del cuaderno es 784 -> Dense 64 relu -> Dense 64 relu -> Dense 10 softmax.
La convolucional es la de la celda 7: Conv2D 32 3x3 same, MaxPool 2x2, Conv2D 64 3x3
same, MaxPool 2x2, Flatten y Dense 10, con nombres de capa propios del cuaderno.
"""

import numpy as np
import pytest
import tensorflow as tf

from tam_mnist.comparacion import (
    NOMBRE_CONVOLUCIONAL,
    NOMBRE_DENSA,
    REDES,
    activation_maps,
    adapt_batch,
    build_conv_network,
    build_dense_network,
    first_conv_filters,
    layer_table,
    load_networks,
    save_network,
)


def _flat_batch(values=None) -> np.ndarray:
    """Un lote de una fila con 784 píxeles, como el que entrega el preprocesamiento."""
    pixels = np.linspace(0, 1, 784, dtype=np.float32) if values is None else values
    return np.asarray(pixels, dtype=np.float32).reshape(1, 784)


# --- Arquitecturas del cuaderno -------------------------------------------------


def test_dense_network_matches_the_notebook():
    """Dos capas ocultas de 64 y una salida de 10: los 55.050 parámetros del cuaderno."""
    model = build_dense_network()
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    assert model.input_shape == (None, 784)
    assert model.output_shape == (None, 10)
    assert [int(layer.units) for layer in dense_layers] == [64, 64, 10]
    assert [layer.activation.__name__ for layer in dense_layers] == ["relu", "relu", "softmax"]
    assert model.count_params() == 784 * 64 + 64 + 64 * 64 + 64 + 64 * 10 + 10


def test_conv_network_matches_the_notebook():
    """Las capas y los nombres de la celda 7 de scratchpad_3, con sus 50.186 parámetros."""
    model = build_conv_network()

    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    assert [layer.name for layer in model.layers] == [
        "Input",
        "Convolucion_1",
        "Pooling_1",
        "Convolucion_2",
        "Pooling_2",
        "Flatten",
        "Densa",
    ]

    convolutions = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    assert [int(layer.filters) for layer in convolutions] == [32, 64]
    assert [tuple(layer.kernel_size) for layer in convolutions] == [(3, 3), (3, 3)]
    assert [layer.padding for layer in convolutions] == ["same", "same"]

    pools = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling2D)]
    assert [tuple(layer.pool_size) for layer in pools] == [(2, 2), (2, 2)]
    assert model.count_params() == 50_186


def test_both_networks_are_registered_with_their_input_shape():
    """El registro dice, para cada red, cómo se llaman y qué forma de lote esperan."""
    assert set(REDES) == {NOMBRE_DENSA, NOMBRE_CONVOLUCIONAL}
    assert REDES[NOMBRE_DENSA].forma_entrada == (784,)
    assert REDES[NOMBRE_CONVOLUCIONAL].forma_entrada == (28, 28, 1)
    assert REDES[NOMBRE_DENSA].titulo and REDES[NOMBRE_CONVOLUCIONAL].titulo


def test_each_network_receives_the_same_pixels_in_its_own_shape():
    """La misma imagen plana llega como vector a la densa y como 28x28x1 a la convolucional."""
    batch = _flat_batch()

    flat = adapt_batch(NOMBRE_DENSA, batch)
    image = adapt_batch(NOMBRE_CONVOLUCIONAL, batch)

    assert flat.shape == (1, 784)
    assert image.shape == (1, 28, 28, 1)
    assert np.array_equal(flat, batch)
    assert np.array_equal(image.reshape(1, 784), batch)


# --- Artefactos de cada red -----------------------------------------------------


def test_artifacts_round_trip_per_network(tmp_path):
    """Guardar y cargar cada red devuelve las mismas predicciones y sus métricas."""
    expectations = {}
    for name, builder in (
        (NOMBRE_DENSA, build_dense_network),
        (NOMBRE_CONVOLUCIONAL, build_conv_network),
    ):
        model = builder()
        batch = adapt_batch(name, _flat_batch())
        expectations[name] = model.predict(batch, verbose=0)
        save_network(name, model, {"test_accuracy": 0.9}, tmp_path)

    loaded = load_networks(tmp_path)

    for name, expected in expectations.items():
        model, metrics = loaded[name]
        batch = adapt_batch(name, _flat_batch())
        assert metrics["test_accuracy"] == pytest.approx(0.9)
        assert np.allclose(model.predict(batch, verbose=0), expected)


def test_loading_without_artifacts_reports_each_network_as_missing(tmp_path):
    """Sin artefactos la app tiene que saber cuáles faltan, sin excepción."""
    loaded = load_networks(tmp_path)

    assert set(loaded) == {NOMBRE_DENSA, NOMBRE_CONVOLUCIONAL}
    assert all(value is None for value in loaded.values())


def test_a_single_missing_artifact_does_not_hide_the_other(tmp_path):
    """Si sólo se entrenó una red, la otra se informa aparte en vez de romper la carga."""
    save_network(NOMBRE_DENSA, build_dense_network(), {"test_accuracy": 0.9}, tmp_path)

    loaded = load_networks(tmp_path)

    assert loaded[NOMBRE_DENSA] is not None
    assert loaded[NOMBRE_CONVOLUCIONAL] is None


# --- Anatomía: tabla, filtros y activaciones ------------------------------------


def test_layer_table_describes_the_convolutional_network():
    """La tabla incluye entrada, convoluciones, pooling, aplanado y salida, con sus formas."""
    table = layer_table(build_conv_network())

    assert [row["Capa"] for row in table] == [
        "entrada",
        "Convolucion_1",
        "Pooling_1",
        "Convolucion_2",
        "Pooling_2",
        "Flatten",
        "Densa",
    ]
    assert [row["Tipo"] for row in table[1:]] == [
        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "MaxPooling2D",
        "Flatten",
        "Dense",
    ]
    assert table[0]["Forma de salida"] == "(None, 28, 28, 1)"
    assert sum(row["Parámetros"] for row in table[1:]) == 50_186


def test_layer_table_still_describes_the_dense_network():
    """La misma tabla sirve para la densa: una fila por capa Dense."""
    table = layer_table(build_dense_network())

    assert [row["Tipo"] for row in table[1:]] == ["Dense", "Dense", "Dense"]
    assert sum(row["Parámetros"] for row in table[1:]) == 55_050


def test_first_convolution_filters_have_one_matrix_per_filter():
    """Los 32 filtros 3x3 del primer canal, listos para dibujar uno por uno."""
    filters = first_conv_filters(build_conv_network())

    assert filters.shape == (32, 3, 3)


def test_activation_maps_keep_the_second_convolution_shape():
    """Los 64 mapas de Convolucion_2 conservan la resolución 14x14 que deja el pooling."""
    maps = activation_maps(build_conv_network(), _flat_batch())

    assert maps.shape == (64, 14, 14)
    assert np.all(np.isfinite(maps))


# --- Script de entrenamiento ----------------------------------------------------


def test_training_script_saves_both_artifacts(tmp_path):
    """Una corrida corta deja las dos redes guardadas y con exactitud de test medida."""
    from entrenar_comparacion import main

    code = main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "1024",
            "--directory",
            str(tmp_path),
            "--verbose",
            "0",
        ]
    )

    assert code == 0
    loaded = load_networks(tmp_path)
    for name, artifact in loaded.items():
        assert artifact is not None, f"falta el artefacto de {name}"
        model, metrics = artifact
        assert 0.0 < metrics["test_accuracy"] <= 1.0
        assert metrics["network"] == name


def test_training_script_can_train_a_single_network(tmp_path):
    """`--redes` permite entrenar una sola red sin esperar la otra."""
    from entrenar_comparacion import main

    code = main(
        [
            "--redes",
            NOMBRE_DENSA,
            "--epochs",
            "1",
            "--batch-size",
            "1024",
            "--directory",
            str(tmp_path),
            "--verbose",
            "0",
        ]
    )

    assert code == 0
    loaded = load_networks(tmp_path)
    assert loaded[NOMBRE_DENSA] is not None
    assert loaded[NOMBRE_CONVOLUCIONAL] is None
