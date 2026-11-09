"""Tests del preprocesamiento de imágenes para la app de la práctica MNIST.

Cubren las reglas que el cuaderno define en `preprocess_uploaded_digit`
(celda 13 de CodeTAM.ipynb) y la conversión del lienzo de dibujo.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from tam_mnist.preproceso import (
    MENSAJE_SIN_DIGITO,
    preprocess_bytes,
    preprocess_canvas,
    preprocess_image,
    predict_probabilities,
)


def make_image(pixels: np.ndarray) -> Image.Image:
    """Construye una imagen PIL en modo L a partir de una matriz de píxeles."""
    return Image.fromarray(pixels.astype(np.uint8), mode="L")


def digit_on_background(background: int, stroke: int, size: int = 100) -> Image.Image:
    """Dibuja un cuadrado central que simula el trazo de un dígito."""
    pixels = np.full((size, size), background, dtype=np.uint8)
    pixels[30:70, 45:55] = stroke
    return make_image(pixels)


def test_blank_image_is_rejected():
    """Una imagen sin trazo no puede predecirse."""
    blank = make_image(np.full((80, 80), 255, dtype=np.uint8))

    with pytest.raises(ValueError) as error:
        preprocess_image(blank)

    assert str(error.value) == MENSAJE_SIN_DIGITO


def test_ink_visible():
    """El mensaje de error es el literal del cuaderno."""
    assert MENSAJE_SIN_DIGITO == "No se encontró un dígito visible en la imagen."


@pytest.mark.parametrize(
    "background, stroke",
    [(255, 0), (0, 255)],
    ids=["fondo_claro", "fondo_oscuro"],
)
def test_both_polarities_produce_the_same_orientation(background, stroke):
    """MNIST usa trazo claro sobre fondo oscuro; el fondo claro se invierte."""
    vector, preview = preprocess_image(digit_on_background(background, stroke))

    assert vector.shape == (1, 784)
    assert vector.dtype == np.float32
    assert preview.shape == (28, 28)
    assert float(preview.min()) >= 0.0 and float(preview.max()) <= 1.0
    assert float(preview.mean()) < 0.5, "el fondo debe seguir siendo oscuro"
    assert float(preview[14, 14]) > 0.5, "el trazo debe quedar claro y centrado"
    assert np.allclose(vector, preview.reshape(1, 784))


def test_stroke_is_cropped_and_scaled():
    """El trazo se recorta y se limita a 20×20, sin importar el tamaño de la foto."""
    pixels = np.zeros((120, 160), dtype=np.uint8)
    pixels[5:15, 5:15] = 255  # trazo chico en la esquina superior izquierda
    _, preview = preprocess_image(make_image(pixels))

    rows, columns = np.where(preview > 0.0)
    assert rows.size > 0
    height = rows.max() - rows.min() + 1
    width = columns.max() - columns.min() + 1
    assert max(height, width) <= 20, "el trazo nunca supera 20 píxeles"


def test_stroke_is_centered_by_mass_like_mnist():
    """El centro de masa del trazo cae en el centro del lienzo de 28×28.

    MNIST normaliza por centro de masa. Centrar por el punto medio de la caja deja
    los dígitos corridos y la red confunde clases parecidas, así que un trazo
    asimétrico tiene que quedar centrado por masa, no por caja.
    """
    pixels = np.zeros((200, 200), dtype=np.uint8)
    pixels[40:100, 40:100] = 255  # bloque grande arriba a la izquierda
    pixels[150:166, 160:176] = 255  # bloque chico abajo a la derecha

    _, preview = preprocess_image(make_image(pixels))

    grid_rows = np.arange(preview.shape[0], dtype=np.float64)
    grid_columns = np.arange(preview.shape[1], dtype=np.float64)
    total = float(preview.sum())
    center_row = float((preview.sum(axis=1) * grid_rows).sum() / total)
    center_column = float((preview.sum(axis=0) * grid_columns).sum() / total)

    assert abs(center_row - 13.5) <= 1.0
    assert abs(center_column - 13.5) <= 1.0

    # La caja completa sigue entera dentro del lienzo: nada se recorta al centrar.
    rows, columns = np.where(preview > 0.0)
    assert rows.min() >= 0 and rows.max() < 28
    assert columns.min() >= 0 and columns.max() < 28


def test_bytes_entry_point_decodes_and_preprocesses():
    """La app recibe bytes desde el uploader, no un objeto PIL."""
    image = digit_on_background(255, 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    from_image = preprocess_image(image)[0]
    from_bytes = preprocess_bytes(buffer.getvalue())[0]

    assert np.allclose(from_image, from_bytes)


def test_bytes_entry_point_rejects_corrupted_data():
    """Un archivo que no es una imagen falla con un error legible."""
    with pytest.raises(ValueError) as error:
        preprocess_bytes(b"esto no es una imagen")

    assert "imagen" in str(error.value).lower()


def test_canvas_rgba_stroke_is_preprocessed_like_a_photo():
    """El lienzo entrega RGBA con trazo opaco sobre fondo transparente."""
    canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    canvas[100:180, 130:150] = [255, 255, 255, 255]

    vector, preview = preprocess_canvas(canvas)

    assert vector.shape == (1, 784)
    assert preview.shape == (28, 28)
    assert float(preview.mean()) < 0.5
    assert float(preview[14, 14]) > 0.5


def test_empty_canvas_is_rejected():
    """Un lienzo vacío equivale a una imagen sin dígito."""
    with pytest.raises(ValueError) as error:
        preprocess_canvas(np.zeros((280, 280, 4), dtype=np.uint8))

    assert str(error.value) == MENSAJE_SIN_DIGITO


class StubModel:
    """Sustituto mínimo del modelo para probar la presentación del resultado."""

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert np.asarray(batch).shape == (1, 784)
        return self._probabilities.reshape(1, 10)


def test_predict_probabilities_returns_ranked_digits():
    """La predicción se presenta ordenada por probabilidad, con la clase ganadora."""
    probabilities = np.zeros(10, dtype=np.float32)
    probabilities[3] = 0.7
    probabilities[7] = 0.2
    probabilities[1] = 0.1

    result = predict_probabilities(StubModel(probabilities), np.zeros((1, 784)))

    assert result.digit == 3
    assert result.confidence == pytest.approx(0.7, abs=1e-6)
    assert [digit for digit, _ in result.ranking] == [3, 7, 1, 0, 2, 4, 5, 6, 8, 9]
    assert sum(probability for _, probability in result.ranking) == pytest.approx(1.0, abs=1e-5)
