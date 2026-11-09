"""Modelo de la práctica MNIST: construcción, entrenamiento y artefacto entrenado.

Replica el experimento del cuaderno `CodeTAM.ipynb`:

* entrada de 784 características (una imagen 28 × 28 aplanada);
* dos capas ocultas `Dense` de 64 y 16 neuronas con `ReLU`;
* salida de 10 unidades con `softmax`, una probabilidad por dígito;
* pérdida `SparseCategoricalCrossentropy` sobre etiquetas enteras;
* Adam, 5 épocas, lotes de 128 y 10 % de los datos apartado para validación.

El artefacto entrenado se guarda en `modelo/mnist_digits.keras` junto con
`modelo/metricas.json`, para que la app arranque sin volver a entrenar.
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_FILE_NAME = "mnist_digits.keras"
METRICS_FILE_NAME = "metricas.json"
MODEL_VERSION = 1
ENV_MODEL_DIR = "TAM_MNIST_MODEL_DIR"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "modelo"

FEATURES = 28 * 28
CLASSES = 10
HIDDEN_UNITS = (64, 16)
SEED = 42

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_VALIDATION_SPLIT = 0.1


def model_directory() -> Path:
    """Carpeta del artefacto, redirigible con `TAM_MNIST_MODEL_DIR`."""
    override = os.environ.get(ENV_MODEL_DIR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_MODEL_DIR


def set_seeds(semilla: int = SEED) -> None:
    """Fija las semillas de NumPy y TensorFlow para que el experimento sea repetible."""
    np.random.seed(semilla)
    tf.random.set_seed(semilla)


def build_model() -> tf.keras.Model:
    """Construye la misma red densa que define el cuaderno."""
    inputs = tf.keras.Input(shape=(FEATURES,), name="pixels")
    hidden_1 = tf.keras.layers.Dense(64, activation="relu", name="hidden_1")(inputs)
    hidden_2 = tf.keras.layers.Dense(16, activation="relu", name="hidden_2")(hidden_1)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="class_probabilities")(hidden_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_digit_classifier")


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compila con la pérdida sparse que corresponde a las etiquetas enteras."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def load_dataset():
    """Descarga (o toma del caché) MNIST y lo deja en la forma que usa el cuaderno."""
    (x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train_raw.reshape((-1, FEATURES)).astype(np.float32) / 255.0
    x_test = x_test_raw.reshape((-1, FEATURES)).astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    semilla: int = SEED,
    verbose: int = 2,
):
    """Entrena el modelo y devuelve `(modelo, historial, metricas)`."""
    set_seeds(semilla)
    (x_train, y_train), (x_test, y_test) = load_dataset()

    model = compile_model(build_model())
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "validation_split": float(validation_split),
        "seed": int(semilla),
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "history": {
            "accuracy": [float(value) for value in history.history["accuracy"]],
            "val_accuracy": [float(value) for value in history.history["val_accuracy"]],
            "loss": [float(value) for value in history.history["loss"]],
            "val_loss": [float(value) for value in history.history["val_loss"]],
        },
    }
    return model, history, metrics


def save_artifact(model: tf.keras.Model, metrics: dict, directory: Path | None = None) -> Path:
    """Guarda el modelo entrenado y sus métricas; devuelve la ruta del artefacto."""
    destination = Path(directory) if directory is not None else model_directory()
    destination.mkdir(parents=True, exist_ok=True)

    model_path = destination / MODEL_FILE_NAME
    model.save(model_path)

    payload = dict(metrics)
    payload["model_version"] = MODEL_VERSION
    (destination / METRICS_FILE_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model_path


def load_artifact(directory: Path | None = None):
    """Carga `(modelo, metricas)`; devuelve `None` si todavía no hay artefacto."""
    source = Path(directory) if directory is not None else model_directory()
    model_path = source / MODEL_FILE_NAME
    if not model_path.is_file():
        return None

    model = tf.keras.models.load_model(model_path)
    metrics_path = source / METRICS_FILE_NAME
    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return model, metrics


def analyze_layer_structure(model: tf.keras.Model) -> list[dict]:
    """Describe la arquitectura como la tabla final del cuaderno."""
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    input_units = int(np.prod(model.input_shape[1:]))

    rows = [
        {
            "Capa": "entrada",
            "Tipo": "Input",
            "Neuronas / unidades": f"{input_units} características",
            "Activación": "Ninguna",
            "Forma de salida": str(model.input_shape),
            "Parámetros": 0,
        }
    ]
    for layer in dense_layers:
        rows.append(
            {
                "Capa": layer.name,
                "Tipo": type(layer).__name__,
                "Neuronas / unidades": int(layer.units),
                "Activación": layer.activation.__name__,
                "Forma de salida": str(tuple(layer.output.shape)),
                "Parámetros": layer.count_params(),
            }
        )
    return rows
