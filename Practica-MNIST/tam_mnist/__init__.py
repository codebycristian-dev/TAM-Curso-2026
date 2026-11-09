"""Núcleo de la práctica MNIST: preprocesamiento de dígitos y modelo entrenado.

Los módulos replican las decisiones del cuaderno `CodeTAM.ipynb` para que la app de
Streamlit y el cuaderno describan exactamente el mismo experimento.
"""

from .preproceso import (
    MENSAJE_SIN_DIGITO,
    Prediction,
    preprocess_bytes,
    preprocess_canvas,
    preprocess_image,
    predict_probabilities,
)

__all__ = [
    "MENSAJE_SIN_DIGITO",
    "Prediction",
    "predict_probabilities",
    "preprocess_bytes",
    "preprocess_canvas",
    "preprocess_image",
]
