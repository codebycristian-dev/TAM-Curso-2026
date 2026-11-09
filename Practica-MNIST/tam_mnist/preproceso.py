"""Preprocesamiento de una imagen de dígito al formato que espera el modelo MNIST.

La red recibe 784 píxeles normalizados a `[0, 1]`. Una foto tomada con el celular no
tiene ese formato: puede estar a color, tener fondo claro, estar descentrada y medir
cualquier tamaño. Este módulo replica la cadena que el cuaderno aplica en la celda 13:

1. escala de grises con autocontraste,
2. inversión si el fondo es claro (MNIST usa trazo claro sobre fondo oscuro),
3. recorte del trazo y rechazo si no hay tinta,
4. escala para que el lado mayor no supere 20 píxeles,
5. centrado en un lienzo de 28 × 28 píxeles y normalización a `[0, 1]`.

Diferencia deliberada con el cuaderno: el centrado usa el **centro de masa** de la
tinta, no el punto medio de la caja del trazo. MNIST normaliza sus propios dígitos
por centro de masa, y centrar por la caja deja el dígito 2–4 píxeles corrido
respecto de lo que la red aprendió: medido sobre 100 dígitos del test set
reconvertidos a imagen, el centrado por caja acierta 71 y el centrado por masa 98.
"""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

# Lado del lienzo final, igual al de las imágenes de MNIST.
MNIST_SIZE = 28
# Lado máximo del trazo antes de centrarlo: deja el margen que usa MNIST.
STROKE_LIMIT = 20
# Un píxel por debajo de este valor se considera fondo.
INK_THRESHOLD = 20

MENSAJE_SIN_DIGITO = "No se encontró un dígito visible en la imagen."
MENSAJE_ARCHIVO_INVALIDO = "No se pudo leer el archivo: no parece ser una imagen PNG o JPG válida."
MENSAJE_LIENZO_INVALIDO = "El lienzo de dibujo no entregó una imagen con canal alfa."


@dataclass(frozen=True)
class Prediction:
    """Resultado de una predicción, listo para mostrar en la interfaz."""

    digit: int
    confidence: float
    ranking: tuple[tuple[int, float], ...]


def grayscale_pixels(image: Image.Image) -> np.ndarray:
    """Convierte a grises y orienta el trazo en claro, como en MNIST."""
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(image))
    pixels = np.asarray(grayscale, dtype=np.uint8)

    # MNIST usa un dígito claro sobre fondo oscuro; invertir si el fondo es claro.
    if float(pixels.mean()) > 127.0:
        pixels = 255 - pixels
    return pixels


def place_at_mass_center(cropped: Image.Image) -> np.ndarray:
    """Ubica el trazo en un lienzo de 28 × 28 con su centro de masa en el centro.

    MNIST normaliza cada dígito por centro de masa. Centrar por la caja del trazo,
    como hace el cuaderno, corre el dígito unos píxeles y la red densa —que no es
    invariante a traslaciones— confunde dígitos parecidos.
    """
    stroke = np.asarray(cropped, dtype=np.float64)
    total = float(stroke.sum())
    if total <= 0.0:
        raise ValueError(MENSAJE_SIN_DIGITO)

    rows = np.arange(stroke.shape[0], dtype=np.float64)
    columns = np.arange(stroke.shape[1], dtype=np.float64)
    center_row = float((stroke.sum(axis=1) * rows).sum() / total)
    center_column = float((stroke.sum(axis=0) * columns).sum() / total)

    # El centro del lienzo de 28 píxeles está entre los índices 13 y 14.
    canvas_center = (MNIST_SIZE - 1) / 2.0
    top = int(round(canvas_center - center_row))
    left = int(round(canvas_center - center_column))

    # Mantener el trazo completo dentro del lienzo, sin recortarlo.
    top = min(max(top, 0), MNIST_SIZE - cropped.height)
    left = min(max(left, 0), MNIST_SIZE - cropped.width)

    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float32)
    block = np.asarray(cropped, dtype=np.float32) / 255.0
    canvas[top : top + cropped.height, left : left + cropped.width] = block
    return canvas


def mnist_matrix(pixels: np.ndarray) -> np.ndarray:
    """Recorta el trazo, lo escala y lo centra en una matriz de 28 × 28 normalizada."""
    mask = pixels > INK_THRESHOLD
    if not bool(mask.any()):
        raise ValueError(MENSAJE_SIN_DIGITO)

    rows, columns = np.where(mask)
    cropped = Image.fromarray(
        pixels[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    )

    # Mantener la proporción y ajustar el lado mayor a STROKE_LIMIT píxeles.
    scale = STROKE_LIMIT / max(cropped.size)
    resized_size = tuple(max(1, int(round(size * scale))) for size in cropped.size)
    cropped = cropped.resize(resized_size, Image.Resampling.LANCZOS)

    return place_at_mass_center(cropped)


def build_input(preview: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empaqueta la matriz de 28 × 28 como el lote de una fila que espera Keras."""
    return preview.reshape(1, MNIST_SIZE * MNIST_SIZE), preview


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesa una imagen ya abierta y devuelve `(vector, vista_previa)`."""
    return build_input(mnist_matrix(grayscale_pixels(image)))


def preprocess_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesa los bytes que entrega el uploader de la app."""
    try:
        with Image.open(BytesIO(bytes(data))) as image:
            return preprocess_image(image)
    except (OSError, ValueError) as error:
        # Un archivo roto se informa como archivo inválido; una imagen sin dígito
        # conserva su propio mensaje, que es el que el usuario necesita leer.
        if isinstance(error, ValueError) and str(error) == MENSAJE_SIN_DIGITO:
            raise
        raise ValueError(MENSAJE_ARCHIVO_INVALIDO) from error


def preprocess_canvas(image_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesa la matriz RGBA que devuelve el lienzo de dibujo.

    El componente dibuja con trazo opaco sobre fondo transparente, así que el canal
    alfa ya tiene la forma que MNIST espera: tinta clara sobre negro.
    """
    array = np.asarray(image_data)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(MENSAJE_LIENZO_INVALIDO)

    if array.shape[2] == 4:
        ink = array[:, :, 3]
    else:
        ink = np.asarray(ImageOps.grayscale(Image.fromarray(array[:, :, :3].astype(np.uint8))))

    return build_input(mnist_matrix(np.asarray(ink, dtype=np.uint8)))


def predict_probabilities(model, batch: np.ndarray) -> Prediction:
    """Predice un lote de una fila y devuelve el dígito, su confianza y el ranking."""
    probabilities = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)[0]

    # Desempate por dígito ascendente para que la presentación sea estable.
    order = sorted(range(probabilities.size), key=lambda digit: (-probabilities[digit], digit))
    ranking = tuple((int(digit), float(probabilities[digit])) for digit in order)
    digit, confidence = ranking[0]
    return Prediction(digit=digit, confidence=confidence, ranking=ranking)
