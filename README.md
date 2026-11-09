# TAM — Curso 2026

Materiales de **Teoría de Aprendizaje de Máquina (2026-2)** de
**Cristian David Dorado Gomez**: los talleres del curso y los parciales resueltos.

La práctica central es la clasificación de dígitos escritos a mano con MNIST: un
cuaderno que entrena una red densa y una app de Streamlit que la usa para predecir
lo que uno escribe.

## Estructura

```
TAM-Curso-2026/
├── Talleres/
│   ├── Practica-MNIST/             práctica de dígitos: cuaderno + app + tests
│   └── Taller-ALternativo-Repaso/  taller de repaso: solución, figuras e informe
├── Parciales/
│   └── Parcial-1/                  parcial resuelto
└── README.md
```

---

## Talleres

### `Talleres/Practica-MNIST/`

| Archivo | Qué es |
| --- | --- |
| `CodeTAM.ipynb` | Cuaderno: prepara MNIST, entrena la red densa, la evalúa y permite probar una imagen propia (en Colab y en VS Code) |
| `app.py` | App de Streamlit: subir una foto, dibujar un dígito con el mouse o elegir una muestra del test set, y ver la predicción |
| `entrenar_modelo.py` | Entrena la red del cuaderno y guarda el artefacto `modelo/mnist_digits.keras` que consume la app |
| `tam_mnist/` | Núcleo reutilizable: preprocesamiento de imágenes (`preproceso.py`) y modelo (`modelo.py`) |
| `tests/` | 25 tests de preprocesamiento, modelo y app, sin navegador |
| `requirements.txt` | Dependencias del cuaderno y de la app |
| `requirements-dev.txt` | Dependencias para correr los tests |

**La app.** Tres entradas contra el mismo modelo entrenado:

| Pestaña | Entrada |
| --- | --- |
| `Subir imagen` | Una o varias fotos o recortes PNG/JPG del dígito hecho a mano |
| `Dibujar` | Un trazo hecho con el mouse en un lienzo |
| `Muestra del test set` | Una imagen de prueba de MNIST que el modelo no entrenó, con la etiqueta real al lado |

El modelo se entrena una sola vez y queda en disco; la app sólo lo carga:

```bash
cd Talleres/Practica-MNIST
source ../../.venv312/bin/activate      # o el entorno con TensorFlow 2.21
python entrenar_modelo.py               # ~12 s: genera modelo/mnist_digits.keras
streamlit run app.py                    # abre la app en el navegador
```

`entrenar_modelo.py` acepta `--epochs`, `--batch-size`, `--validation-split`,
`--seed` y `--directory` (ver `python entrenar_modelo.py --help`). El artefacto
queda en `modelo/`, ignorado por Git porque se regenera con el script.

**Tests:**

```bash
cd Talleres/Practica-MNIST
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

**El cuaderno.** Abrir `CodeTAM.ipynb` y ejecutar las celdas en orden. El
directorio de trabajo debe ser `Talleres/Practica-MNIST/`, porque el cuaderno
resuelve rutas relativas a esa carpeta. La primera ejecución descarga MNIST y
necesita conexión a Internet; después queda en caché.

**Dos diferencias deliberadas de la app con el cuaderno:**

1. Al preparar una imagen externa, la app centra el trazo por su **centro de
   masa**, que es como MNIST normaliza sus propios dígitos, en lugar de por el
   punto medio de la caja del trazo. Con el centrado del cuaderno una foto del
   mismo dígito se predice bien 71 de cada 100 veces; con el centrado por masa, 95.
2. El umbral que separa tinta de papel es **local y adaptativo** en vez de un valor
   fijo sobre la imagen completa. Con un umbral fijo, el ruido del sensor y la
   sombra del papel se detectan como tinta y el dígito se achica hasta desaparecer.

El modelo es exactamente el mismo en los dos casos: 784 entradas, capas densas de
64 y 16 neuronas, salida `softmax` de 10 clases, Adam y 5 épocas con lotes de 128.

### `Talleres/Taller-ALternativo-Repaso/`

| Ruta | Qué es |
| --- | --- |
| `Taller_Alternativo_TAM_2026_2.pdf` | Enunciado del taller |
| `solucion/solucion_taller.ipynb` | Cuaderno de la solución, ya ejecutado con sus salidas |
| `solucion/Taller_Alternativo_TAM_2026_2_solucion.pdf` | Solución en PDF |
| `figuras/` | Figuras producidas por el cuaderno |
| `informe/informe.tex` | Informe en LaTeX |
| `resultados.json` | Resumen numérico exportado por el cuaderno |
| `scripts/generar_notebook.py` | Regenera `solucion/solucion_taller.ipynb` |
| `scripts/ejecutar_notebook.py` | Ejecuta el cuaderno de punta a punta y guarda las salidas |
| `requirements.txt` | Dependencias de Python del taller |

```bash
cd Talleres/Taller-ALternativo-Repaso
python -m pip install -r requirements.txt
python scripts/generar_notebook.py    # escribe solucion/solucion_taller.ipynb
python scripts/ejecutar_notebook.py   # lo ejecuta y actualiza sus salidas
```

`informe.tex` necesita una instalación de LaTeX (`pdflatex`) para compilarse. Ambos
scripts se ejecutan desde esa carpeta: las figuras y `resultados.json` se escriben ahí.

---

## Parciales

### `Parciales/Parcial-1/`

| Archivo | Qué es |
| --- | --- |
| `Solución_Parcial-1_TAM_2026 pdf` | Parcial 1 resuelto, en PDF |
| `Parcial_1_TAM_2026_2_Pearson_RKHS.pdf` | Material del parcial 1 sobre el coeficiente de Pearson y RKHS, en PDF |

---

## Entorno de Python

TensorFlow 2.21 —el que fija `requirements.txt`— publica ruedas para Python
3.10–3.13. Un Python 3.14 del sistema no sirve, y en Ubuntu 26.04 sólo está esa
versión. La ruta sin `sudo` es `uv`, que descarga un CPython 3.12 propio y lo usa
sólo para este proyecto:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/TAM-Curso-2026
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r Talleres/Practica-MNIST/requirements.txt
```

`.venv312/` es el entorno de la práctica de MNIST. El taller de repaso usa el
`.venv` de la raíz, con `numpy`, `matplotlib` y `scikit-learn`.
