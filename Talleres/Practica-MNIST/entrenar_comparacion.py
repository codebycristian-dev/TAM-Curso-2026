"""Entrena las dos redes de comparación de la práctica y guarda sus artefactos.

Son las redes del cuaderno `experimentos/scratchpad_3.ipynb`: la densa 784 → 64 → 64
→ 10 y la convolucional con dos convoluciones y dos pooling. La app las muestra por
separado y las compara; este script es el único lugar donde se entrenan.

Uso:
    python entrenar_comparacion.py                      # las dos redes, 5 épocas
    python entrenar_comparacion.py --redes densa        # sólo la densa
    python entrenar_comparacion.py --epochs 2 --verbose 1
    python entrenar_comparacion.py --directory /tmp/modelo

Los artefactos quedan en `modelo/comparacion/`, aparte del artefacto de la red densa
64-16 que consume el resto de la app (`entrenar_modelo.py`).
"""

import argparse
from pathlib import Path

from tam_mnist.comparacion import (
    NOMBRE_CONVOLUCIONAL,
    NOMBRE_DENSA,
    REDES,
    comparison_directory,
    save_network,
    train_network,
)
from tam_mnist.modelo import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_VALIDATION_SPLIT, SEED

REDES_ENTRENABLES = (NOMBRE_DENSA, NOMBRE_CONVOLUCIONAL)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena la red densa y la red convolucional de comparación, "
            "y guarda sus artefactos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--redes",
        choices=(*REDES_ENTRENABLES, "ambas"),
        default="ambas",
        help="Qué red entrenar; `ambas` entrena la densa y después la convolucional.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Cantidad de épocas de entrenamiento por red.",
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
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Cuánto imprime Keras durante el entrenamiento.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Carpeta del modelo; por defecto modelo/ junto a este script.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    selected = REDES_ENTRENABLES if args.redes == "ambas" else (args.redes,)

    print(f"Carpeta de artefactos: {comparison_directory(args.directory)}")
    print()

    for name in selected:
        print(f"== {REDES[name].titulo} ==")
        print(
            f"Entrenando {args.epochs} épocas (lote {args.batch_size}, "
            f"semilla {args.seed})…"
        )
        model, metrics = train_network(
            name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            semilla=args.seed,
            verbose=args.verbose,
        )
        path = save_network(name, model, metrics, args.directory)
        print(f"Exactitud en prueba: {metrics['test_accuracy']:.2%}")
        print(f"Pérdida en prueba:   {metrics['test_loss']:.4f}")
        print(f"Parámetros:          {metrics['parameters']:,}")
        print(f"Artefacto:           {path}")
        print()

    print("Abrí la app con: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
