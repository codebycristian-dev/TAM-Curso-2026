# TAM — Curso 2026

Materiales de **Teoría de Aprendizaje de Máquina (2026-2)**: la práctica de
clasificación de dígitos escritos a mano con MNIST y el taller de repaso con su
informe.

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `Practica-MNIST/` | Cuaderno de la práctica, app de Streamlit, enunciado de la tarea y sus dependencias |
| `Taller-Repaso/` | Enunciado, solución, figuras, informe y scripts del taller |
| `odd/tasks/` | Documentos de trabajo de cada feature: specs, tareas y bitácora |

### `Practica-MNIST/`

| Archivo | Qué es |
| --- | --- |
| `CodeTAM.ipynb` | Prepara MNIST, entrena una red densa, la evalúa y permite probar una imagen propia (en Colab y en VS Code) |
| `app.py` | App de Streamlit: subir una imagen, dibujar un dígito o elegir una muestra del test set, y ver la predicción del modelo |
| `entrenar_modelo.py` | Entrena la red del cuaderno y guarda el artefacto `modelo/mnist_digits.keras` que consume la app |
| `tam_mnist/` | Núcleo reutilizable: preprocesamiento de imágenes y modelo (`build_model`, `train_model`, artefacto) |
| `tests/` | Tests de preprocesamiento, modelo y app (`pytest`, sin navegador) |
| `TAREA-TAM.txt` | Enunciado de la tarea: interfaz para cargar un dígito escrito a mano y predecirlo con el modelo del cuaderno |
| `requirements.txt` | Dependencias de Python del cuaderno y de la app |
| `requirements-dev.txt` | Dependencias para correr los tests |

Ejecutar el cuaderno:

```bash
cd Practica-MNIST
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternativa sin `sudo` cuando el sistema sólo trae una versión de Python
incompatible con TensorFlow (por ejemplo Python 3.14 en Ubuntu 26.04). `uv`
descarga un CPython 3.12 propio y lo usa sólo para este proyecto:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/TAM-Curso-2026
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r Practica-MNIST/requirements.txt
```

Luego, en VS Code, elegí como kernel el intérprete
`.venv312/bin/python` (Python 3.12.16). El `.venv` de la raíz sigue siendo el
del taller y no se toca.

Después abrir `CodeTAM.ipynb` y ejecutar las celdas en orden. El directorio de
trabajo debe ser `Practica-MNIST/`, porque el cuaderno instala desde
`requirements.txt` y resuelve rutas relativas a esa carpeta. La primera
ejecución descarga MNIST y necesita conexión a Internet.

#### La misma práctica como app de Streamlit

La app hace lo que pide `TAREA-TAM.txt` y agrega dos formas de entrada más, todas
contra el mismo modelo del cuaderno:

| Pestaña | Entrada |
| --- | --- |
| `Subir imagen` | Una o varias fotos o recortes PNG/JPG del dígito |
| `Dibujar` | Un trazo hecho con el mouse en un lienzo |
| `Muestra del test set` | Una imagen de prueba de MNIST que el modelo no entrenó, con la etiqueta real al lado |

El modelo se entrena una sola vez y queda en disco; la app sólo lo carga:

```bash
cd Practica-MNIST
source ../.venv312/bin/activate       # o el entorno con TensorFlow 2.21
python entrenar_modelo.py             # ~1 min: genera modelo/mnist_digits.keras
streamlit run app.py                  # abre la app en el navegador
```

`entrenar_modelo.py` acepta `--epochs`, `--batch-size`, `--validation-split`,
`--seed` y `--directory` (ver `python entrenar_modelo.py --help`). El artefacto
queda en `modelo/`, que está ignorado por Git porque se regenera con el script.

Una diferencia deliberada con el cuaderno: al preparar una imagen externa, la app
centra el trazo por su **centro de masa**, que es como MNIST normaliza sus propios
dígitos, en lugar de por el punto medio de la caja del trazo como hace la celda 13
del cuaderno. Con el centrado del cuaderno, una foto del mismo dígito se predice
bien 71 de cada 100 veces; con el centrado por masa, 95. El modelo es exactamente
el mismo.

Tests:

```bash
cd Practica-MNIST
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

### `Taller-Repaso/`

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

Regenerar y ejecutar la solución:

```bash
cd Taller-Repaso
python -m pip install -r requirements.txt
python scripts/generar_notebook.py    # escribe solucion/solucion_taller.ipynb
python scripts/ejecutar_notebook.py   # lo ejecuta y actualiza sus salidas
```

`informe.tex` necesita una instalación de LaTeX (`pdflatex`) para compilarse.
Ambos scripts se ejecutan desde `Taller-Repaso/`: las figuras y
`resultados.json` se escriben ahí.
