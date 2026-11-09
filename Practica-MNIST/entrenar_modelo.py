"""Entrena la red de la práctica MNIST y guarda el artefacto que consume la app.

Uso:
    python entrenar_modelo.py                 # 5 épocas, guarda en modelo/
    python entrenar_modelo.py --epochs 2      # corrida corta para probar
    python entrenar_modelo.py --directory /tmp/modelo

El experimento es el del cuaderno `CodeTAM.ipynb`: 784 entradas, capas densas de
64 y 16 neuronas, salida softmax de 10 clases, Adam y 5 épocas con lotes de 128.
"""

import argparse
from pathlib import Path

from tam_mnist.modelo import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_VALIDATION_SPLIT,
    MODEL_FILE_NAME,
    SEED,
    model_directory,
    save_artifact,
    train_model,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el clasificador de dígitos MNIST de la práctica y guarda el artefacto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Cantidad de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Tamaño del lote de entrenamiento.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fracción de los datos de entrenamiento reservada para validación.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla de NumPy y TensorFlow para que el resultado sea repetible.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Carpeta donde guardar el modelo; por defecto modelo/ junto a este script.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    destination = args.directory or model_directory()

    print(f"Entrenando {args.epochs} épocas (lote {args.batch_size}, semilla {args.seed})…")
    model, _history, metrics = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        semilla=args.seed,
    )

    model_path = save_artifact(model, metrics, destination)
    print()
    print(f"Exactitud en prueba: {metrics['test_accuracy']:.2%}")
    print(f"Pérdida en prueba:   {metrics['test_loss']:.4f}")
    print(f"Artefacto:           {model_path}")
    print(f"Archivo del modelo:  {MODEL_FILE_NAME}")
    print()
    print("Abrí la app con: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
