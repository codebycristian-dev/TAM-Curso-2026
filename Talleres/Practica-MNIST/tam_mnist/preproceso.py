"""Preprocesamiento de una imagen de dígito al formato que espera el modelo MNIST.

La red recibe 784 píxeles normalizados a `[0, 1]`. Una foto de un dígito hecho con
lapicero sobre una hoja no tiene ese formato: puede estar a color, tener sombras,
ruido del sensor, el escritorio alrededor de la hoja y medir cualquier tamaño.

La cadena es:

1. escala de grises y estimación del fondo local con un desenfoque amplio,
2. separación del trazo respecto de ese fondo local, eligiendo la polaridad por
   masa (sirve tanto lapicero oscuro sobre papel como trazo claro sobre negro),
3. binarización con el umbral de Otsu más un piso absoluto de tinta, que descarta
   el ruido del sensor,
4. quedar con el trazo conectado más grande y descartar el resto,
5. recorte, escala para que el lado mayor no supere 20 píxeles,
6. centrado por **centro de masa** en un lienzo de 28 × 28 y normalización a `[0, 1]`.

Diferencias deliberadas con el cuaderno `CodeTAM.ipynb` (celda 13):

* El centrado usa el centro de masa del trazo, no el punto medio de la caja. MNIST
  normaliza sus dígitos por centro de masa y el corrimiento de 2–4 píxeles alcanza
  para que la red densa confunda un 7 con un 2.
* El umbral es local y adaptativo, no un valor fijo sobre la imagen completa. Con
  un umbral fijo, el ruido del sensor y la sombra del papel se detectan como tinta,
  el "trazo" pasa a ser la foto entera y el dígito se achica hasta desaparecer.
"""

from collections import deque
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Lado del lienzo final, igual al de las imágenes de MNIST.
MNIST_SIZE = 28
# Lado máximo del trazo antes de centrarlo: deja el margen que usa MNIST.
STROKE_LIMIT = 20
# Diferencia mínima de gris, respecto del fondo local, para considerar tinta.
# Por debajo de esto sólo hay ruido del sensor.
MIN_INK = 30.0
# Fracción mínima del área de la imagen que debe ocupar el trazo elegido.
MIN_BLOB_FRACTION = 0.0008

MENSAJE_SIN_DIGITO = "No se encontró un dígito visible en la imagen."
MENSAJE_ARCHIVO_INVALIDO = "No se pudo leer el archivo: no parece ser una imagen PNG o JPG válida."
MENSAJE_LIENZO_INVALIDO = "El lienzo de dibujo no entregó una imagen con canal alfa."


@dataclass(frozen=True)
class Prediction:
    """Resultado de una predicción, listo para mostrar en la interfaz."""

    digit: int
    confidence: float
    ranking: tuple[tuple[int, float], ...]


def background_radius(size: tuple[int, int]) -> float:
    """Radio del desenfoque que estima la iluminación del papel.

    Chico en imágenes chicas (un dígito de 28 píxeles) y amplio en fotos, donde la
    sombra varía en distancias grandes. Cuando el trazo es más ancho que la ventana,
    su centro se estima como fondo y el trazo queda hueco: de eso se ocupa
    `fill_holes`.
    """
    return float(min(max(min(size) / 12.0, 2.0), 60.0))


def ink_map(image: Image.Image) -> np.ndarray:
    """Devuelve qué tan lejos está cada píxel del fondo local, en unidades de gris.

    Compara contra un desenfoque amplio en lugar del brillo global, así una sombra
    sobre el papel o una foto subexpuesta no cambian el resultado. La polaridad se
    elige por masa: gana la hipótesis con más trazo.
    """
    gray = ImageOps.grayscale(image)
    pixels = np.asarray(gray, dtype=np.float32)
    background = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(background_radius(gray.size))), dtype=np.float32
    )

    darker = np.clip(background - pixels, 0.0, None)  # lapicero sobre papel
    lighter = np.clip(pixels - background, 0.0, None)  # trazo claro sobre negro
    return darker if float(darker.sum()) >= float(lighter.sum()) else lighter


def otsu_threshold(values: np.ndarray) -> float:
    """Umbral de Otsu: separa dos poblaciones de intensidad maximizando la varianza."""
    flat = np.asarray(values, dtype=np.float32).ravel()
    histogram, edges = np.histogram(flat, bins=256, range=(0.0, 256.0))
    centers = (edges[:-1] + edges[1:]) / 2.0

    weight_background = np.cumsum(histogram).astype(np.float64)
    weight_foreground = weight_background[-1] - weight_background
    total = weight_background[-1]
    if total <= 0:
        return 0.0

    cumulative_mass = np.cumsum(histogram * centers).astype(np.float64)
    cumulative_total = cumulative_mass[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_background = cumulative_mass / weight_background
        mean_foreground = (cumulative_total - cumulative_mass) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

    between = np.nan_to_num(between, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centers[int(np.argmax(between))])


def _downsample(mask: np.ndarray, factor: int, pool=np.any) -> np.ndarray:
    """Reduce la imagen agrupando bloques con la función `pool`."""
    if factor <= 1:
        return mask
    rows, columns = mask.shape
    pad_rows = (-rows) % factor
    pad_columns = (-columns) % factor
    padded = np.pad(mask, ((0, pad_rows), (0, pad_columns)), constant_values=0)
    return pool(
        padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor),
        axis=(1, 3),
    )


def _upsample(mask: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    """Devuelve la máscara reducida al tamaño original."""
    if factor <= 1:
        return mask[: shape[0], : shape[1]]
    grown = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    return grown[: shape[0], : shape[1]]


def heaviest_blob(mask: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Se queda con el conjunto de tinta conectado que tiene más tinta.

    El puntaje es la cantidad de tinta y no el área: el borde del escritorio o el
    marco de la hoja pueden tener más píxeles que el dígito pero mucha menos tinta.
    Además se prefiere un trazo que no toque el borde de la foto, porque el dígito
    está adentro y lo que toca el borde es el fondo.

    El etiquetado corre sobre una versión reducida de la máscara para que sea rápido
    incluso con fotos grandes, y después se aplica al tamaño original.
    """
    if not bool(mask.any()):
        return mask

    factor = max(1, int(min(mask.shape) // 160))
    small = _downsample(mask, factor)
    small_ink = _downsample(ink, factor, pool=np.sum)
    rows, columns = small.shape
    visited = np.zeros_like(small, dtype=bool)

    best_mass = -1.0
    best: list[tuple[int, int]] = []
    best_interior_mass = -1.0
    best_interior: list[tuple[int, int]] = []

    for start_row in range(rows):
        row = small[start_row]
        for start_column in np.flatnonzero(row & ~visited[start_row]):
            if visited[start_row, start_column]:
                continue
            component: list[tuple[int, int]] = []
            touches_border = False
            queue = deque([(start_row, int(start_column))])
            visited[start_row, start_column] = True
            while queue:
                current_row, current_column = queue.popleft()
                component.append((current_row, current_column))
                if (
                    current_row in (0, rows - 1)
                    or current_column in (0, columns - 1)
                ):
                    touches_border = True
                for delta_row, delta_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbour_row = current_row + delta_row
                    neighbour_column = current_column + delta_column
                    if (
                        0 <= neighbour_row < rows
                        and 0 <= neighbour_column < columns
                        and small[neighbour_row, neighbour_column]
                        and not visited[neighbour_row, neighbour_column]
                    ):
                        visited[neighbour_row, neighbour_column] = True
                        queue.append((neighbour_row, neighbour_column))

            mass = float(sum(small_ink[row_index, column_index] for row_index, column_index in component))
            if mass > best_mass:
                best_mass, best = mass, component
            if not touches_border and mass > best_interior_mass:
                best_interior_mass, best_interior = mass, component

    chosen = best_interior or best
    if not chosen:
        return np.zeros_like(mask)

    chosen_small = np.zeros_like(small, dtype=bool)
    for row_index, column_index in chosen:
        chosen_small[row_index, column_index] = True
    return mask & _upsample(chosen_small, factor, mask.shape)


def fill_holes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rellena los huecos internos del trazo y devuelve `(máscara_llena, huecos)`.

    Si el trazo es más ancho que la ventana del fondo local, su centro se parece al
    papel y queda hueco. Un dígito es una mancha sólida: lo que no se alcanza desde
    el borde de su caja es interior. El relleno se calcula sólo dentro de la caja del
    trazo, no en toda la foto.
    """
    rows, columns = np.where(mask)
    if rows.size == 0:
        return mask, mask

    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    window = mask[top:bottom, left:right]
    height, width = window.shape

    reachable = np.zeros_like(window, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(row: int, column: int) -> None:
        if not window[row, column] and not reachable[row, column]:
            reachable[row, column] = True
            queue.append((row, column))

    for column in range(width):
        push(0, column)
        push(height - 1, column)
    for row in range(height):
        push(row, 0)
        push(row, width - 1)

    while queue:
        current_row, current_column = queue.popleft()
        for delta_row, delta_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour_row = current_row + delta_row
            neighbour_column = current_column + delta_column
            if 0 <= neighbour_row < height and 0 <= neighbour_column < width:
                push(neighbour_row, neighbour_column)

    holes_window = ~window & ~reachable
    holes = np.zeros_like(mask, dtype=bool)
    holes[top:bottom, left:right] = holes_window

    filled = mask.copy()
    filled[top:bottom, left:right] = window | holes_window
    return filled, holes


def stroke_mask(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binariza el mapa de tinta, se queda con el trazo más grande y lo rellena."""
    threshold = max(otsu_threshold(ink), MIN_INK)
    mask = ink >= threshold
    if not bool(mask.any()):
        raise ValueError(MENSAJE_SIN_DIGITO)

    mask = heaviest_blob(mask, ink)
    if int(mask.sum()) < max(20, int(MIN_BLOB_FRACTION * mask.size)):
        raise ValueError(MENSAJE_SIN_DIGITO)
    return fill_holes(mask)


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


def build_input(preview: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empaqueta la matriz de 28 × 28 como el lote de una fila que espera Keras."""
    return preview.reshape(1, MNIST_SIZE * MNIST_SIZE), preview


def matrix_from_ink(ink: np.ndarray) -> np.ndarray:
    """Recorta el trazo del mapa de tinta, lo escala y lo centra en 28 × 28."""
    mask, holes = stroke_mask(ink)
    rows, columns = np.where(mask)
    window = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
    cropped = np.array(ink[window], dtype=np.float64)

    # Llevar el trazo a rango completo sin que un píxel suelto defina la escala.
    high = float(np.percentile(cropped, 99.5))
    top_value = max(high, 0.5 * float(cropped.max()), 1.0)

    # El interior del trazo es tinta, no papel: el relleno de huecos también se
    # aplica a la intensidad, para que el trazo no llegue hueco a la red.
    cropped[holes[window]] = top_value

    normalized = Image.fromarray(np.clip(cropped * (255.0 / top_value), 0, 255).astype(np.uint8))

    # Mantener la proporción y ajustar el lado mayor a STROKE_LIMIT píxeles.
    scale = STROKE_LIMIT / max(normalized.size)
    resized_size = tuple(max(1, int(round(size * scale))) for size in normalized.size)
    normalized = normalized.resize(resized_size, Image.Resampling.LANCZOS)

    return place_at_mass_center(normalized)


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesa una imagen ya abierta y devuelve `(vector, vista_previa)`."""
    return build_input(matrix_from_ink(ink_map(image)))


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
    alfa ya es exactamente el trazo: no hace falta estimar fondo ni polaridad.
    """
    array = np.asarray(image_data)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(MENSAJE_LIENZO_INVALIDO)

    if array.shape[2] == 4:
        alpha = np.asarray(array[:, :, 3], dtype=np.float32)
    else:
        alpha = np.asarray(
            ImageOps.grayscale(Image.fromarray(array[:, :, :3].astype(np.uint8))), dtype=np.float32
        )

    return build_input(matrix_from_ink(alpha))


def predict_probabilities(model, batch: np.ndarray) -> Prediction:
    """Predice un lote de una fila y devuelve el dígito, su confianza y el ranking."""
    probabilities = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)[0]

    # Desempate por dígito ascendente para que la presentación sea estable.
    order = sorted(range(probabilities.size), key=lambda digit: (-probabilities[digit], digit))
    ranking = tuple((int(digit), float(probabilities[digit])) for digit in order)
    digit, confidence = ranking[0]
    return Prediction(digit=digit, confidence=confidence, ranking=ranking)
