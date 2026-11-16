"""Las dos redes del cuaderno `experimentos/scratchpad_3.ipynb`, lado a lado.

El cuaderno de clase compara una red densa con una convolucional sobre el mismo MNIST.
Este módulo replica esas dos arquitecturas para que la app pueda mostrarlas por
separado y compararlas:

* `densa`: 784 entradas, dos capas ocultas de 64 neuronas y salida `softmax` de 10
  clases. 55.050 parámetros.
* `convolucional`: entrada 28 × 28 × 1, `Conv2D` de 32 filtros 3 × 3 con `padding`
  `same`, `MaxPooling2D` 2 × 2, `Conv2D` de 64 filtros 3 × 3, otro `MaxPooling2D`,
  `Flatten` y la salida de 10 clases. 50.186 parámetros: ocho veces menos que la
  convolucional sin pooling del cuaderno `scratchpad_2.ipynb`, y con mejor exactitud.

Los nombres de capa de la convolucional son los del cuaderno, porque la vista de
anatomía busca `Convolucion_1` y `Convolucion_2` por nombre para dibujar los filtros y
los mapas de activación.

Cada red se guarda en su propio artefacto, dentro de `modelo/comparacion/`, para no
tocar el artefacto de la red densa 64-16 que usa el resto de la app.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from .modelo import CLASSES, FEATURES, METRICS_FILE_NAME, MODEL_VERSION, compile_model, model_directory

NOMBRE_DENSA = "densa"
NOMBRE_CONVOLUCIONAL = "convolucional"
COMPARISON_DIR_NAME = "comparacion"

# Lado de la imagen de MNIST antes de aplanarla, y lado que deja el pooling 2 × 2.
IMAGE_SIDE = 28
CHANNELS = 1


@dataclass(frozen=True)
class RedSpec:
    """Todo lo que la app necesita saber de una de las dos redes."""

    nombre: str
    titulo: str
    constructor: Callable[[], tf.keras.Model]
    forma_entrada: tuple[int, ...]
    archivo: str


def build_dense_network() -> tf.keras.Model:
    """La red densa del cuaderno: 784 → 64 → 64 → 10."""
    inputs = tf.keras.Input(shape=(FEATURES,), name="pixels")
    hidden_1 = tf.keras.layers.Dense(64, activation="relu", name="densa_1")(inputs)
    hidden_2 = tf.keras.layers.Dense(64, activation="relu", name="densa_2")(hidden_1)
    outputs = tf.keras.layers.Dense(CLASSES, activation="softmax", name="class_probabilities")(
        hidden_2
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_dense_64x64")


def build_conv_network() -> tf.keras.Model:
    """La red convolucional de la celda 7 del cuaderno, con sus mismos nombres de capa."""
    inputs = tf.keras.Input(shape=(IMAGE_SIDE, IMAGE_SIDE, CHANNELS), name="Input")
    conv_1 = tf.keras.layers.Conv2D(
        32, kernel_size=3, activation="relu", strides=1, padding="same", name="Convolucion_1"
    )(inputs)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="Pooling_1")(conv_1)
    conv_2 = tf.keras.layers.Conv2D(
        64, kernel_size=3, activation="relu", strides=1, padding="same", name="Convolucion_2"
    )(pool_1)
    pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="Pooling_2")(conv_2)
    flattened = tf.keras.layers.Flatten(name="Flatten")(pool_2)
    outputs = tf.keras.layers.Dense(CLASSES, activation="softmax", name="Densa")(flattened)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_convolutional")


REDES: dict[str, RedSpec] = {
    NOMBRE_DENSA: RedSpec(
        nombre=NOMBRE_DENSA,
        titulo="Red densa 64-64",
        constructor=build_dense_network,
        forma_entrada=(FEATURES,),
        archivo="densa_64x64.keras",
    ),
    NOMBRE_CONVOLUCIONAL: RedSpec(
        nombre=NOMBRE_CONVOLUCIONAL,
        titulo="Red convolucional",
        constructor=build_conv_network,
        forma_entrada=(IMAGE_SIDE, IMAGE_SIDE, CHANNELS),
        archivo="convolucional.keras",
    ),
}


def comparison_directory(directory: Path | None = None) -> Path:
    """Carpeta de los dos artefactos nuevos, dentro del directorio del modelo."""
    base = Path(directory) if directory is not None else model_directory()
    return base / COMPARISON_DIR_NAME


def adapt_batch(nombre: str, batch: np.ndarray) -> np.ndarray:
    """Lleva un lote de píxeles planos a la forma que espera la red pedida.

    El preprocesamiento entrega siempre `(1, 784)`. La densa lo usa tal cual y la
    convolucional necesita `(1, 28, 28, 1)` con los mismos píxeles.
    """
    spec = REDES[nombre]
    array = np.asarray(batch, dtype=np.float32)
    return array.reshape((array.shape[0],) + spec.forma_entrada)


def save_network(
    nombre: str, model: tf.keras.Model, metrics: dict, directory: Path | None = None
) -> Path:
    """Guarda el artefacto de una red y sus métricas; devuelve la ruta del modelo."""
    spec = REDES[nombre]
    destination = comparison_directory(directory)
    destination.mkdir(parents=True, exist_ok=True)

    model_path = destination / spec.archivo
    model.save(model_path)

    payload = dict(metrics)
    payload["model_version"] = MODEL_VERSION
    payload["network"] = nombre
    (destination / f"{model_path.stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model_path


def load_networks(directory: Path | None = None) -> dict[str, tuple[tf.keras.Model, dict] | None]:
    """Carga las dos redes; cada una vale `None` si su artefacto todavía no existe."""
    source = comparison_directory(directory)
    loaded: dict[str, tuple[tf.keras.Model, dict] | None] = {}

    for nombre, spec in REDES.items():
        model_path = source / spec.archivo
        if not model_path.is_file():
            loaded[nombre] = None
            continue

        model = tf.keras.models.load_model(model_path)
        metrics_path = source / f"{model_path.stem}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        loaded[nombre] = (model, metrics)

    return loaded


def train_network(
    nombre: str,
    epochs: int,
    batch_size: int,
    validation_split: float,
    semilla: int,
    verbose: int = 2,
):
    """Entrena una de las dos redes y devuelve `(modelo, métricas)` con la exactitud en test."""
    from .modelo import load_dataset, set_seeds

    spec = REDES[nombre]
    set_seeds(semilla)
    (x_train, y_train), (x_test, y_test) = load_dataset()

    train_batches = adapt_batch(nombre, x_train)
    test_batches = adapt_batch(nombre, x_test)

    model = compile_model(spec.constructor())
    history = model.fit(
        train_batches,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )

    test_loss, test_accuracy = model.evaluate(test_batches, y_test, verbose=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "validation_split": float(validation_split),
        "seed": int(semilla),
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "parameters": int(model.count_params()),
        "history": {
            "accuracy": [float(value) for value in history.history["accuracy"]],
            "val_accuracy": [float(value) for value in history.history["val_accuracy"]],
            "loss": [float(value) for value in history.history["loss"]],
            "val_loss": [float(value) for value in history.history["val_loss"]],
        },
    }
    return model, metrics


def _units(layer: tf.keras.layers.Layer) -> int | str:
    """La columna "unidades" de la tabla, según el tipo de capa."""
    if isinstance(layer, tf.keras.layers.Dense):
        return int(layer.units)
    if isinstance(layer, tf.keras.layers.Conv2D):
        height, width = layer.kernel_size
        return f"{int(layer.filters)} filtros {height}×{width}"
    if isinstance(layer, tf.keras.layers.MaxPooling2D):
        height, width = layer.pool_size
        return f"pool {height}×{width}"
    return "—"


def _activation(layer: tf.keras.layers.Layer) -> str:
    """La activación de la capa, o "Ninguna" cuando no tiene."""
    activation = getattr(layer, "activation", None)
    return activation.__name__ if activation is not None else "Ninguna"


def layer_table(model: tf.keras.Model) -> list[dict]:
    """Describe la arquitectura completa, incluidas convoluciones, pooling y aplanado."""
    rows = [
        {
            "Capa": "entrada",
            "Tipo": "Input",
            "Neuronas / unidades": f"{int(np.prod(model.input_shape[1:]))} características",
            "Activación": "Ninguna",
            "Forma de salida": str(model.input_shape),
            "Parámetros": 0,
        }
    ]
    for layer in model.layers:
        # Keras incluye la capa de entrada en `model.layers`; la fila "entrada" ya la
        # describe arriba, así que acá se saltea para no repetirla.
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        rows.append(
            {
                "Capa": layer.name,
                "Tipo": type(layer).__name__,
                "Neuronas / unidades": _units(layer),
                "Activación": _activation(layer),
                "Forma de salida": str(tuple(layer.output.shape)),
                "Parámetros": layer.count_params(),
            }
        )
    return rows


def _convolutions(model: tf.keras.Model) -> list[tf.keras.layers.Conv2D]:
    """Las capas convolucionales del modelo, en orden."""
    return [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]


def first_conv_filters(model: tf.keras.Model) -> np.ndarray:
    """Los filtros de `Convolucion_1` como `(filtros, alto, ancho)`, uno por matriz."""
    layers = _convolutions(model)
    if not layers:
        raise ValueError("El modelo no tiene capas convolucionales.")

    weights = np.asarray(layers[0].get_weights()[0], dtype=np.float32)
    # Keras guarda (alto, ancho, canales de entrada, filtros); con una sola entrada
    # queda un mapa 2D por filtro, que es lo que se dibuja.
    return np.moveaxis(weights, -1, 0)[..., 0]


def activation_maps(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """Los mapas de activación de la última convolución, como `(filtros, alto, ancho)`."""
    layers = _convolutions(model)
    if not layers:
        raise ValueError("El modelo no tiene capas convolucionales.")

    probe = tf.keras.Model(inputs=model.input, outputs=layers[-1].output)
    maps = np.asarray(probe.predict(adapt_batch(NOMBRE_CONVOLUCIONAL, batch), verbose=0))[0]
    return np.moveaxis(maps, -1, 0)
