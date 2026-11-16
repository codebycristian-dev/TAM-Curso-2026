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
| `app.py` | App de Streamlit: subir una foto del dígito y ver la predicción. Además muestra las dos redes del cuaderno de clase, cada una con su anatomía, y las compara |
| `entrenar_modelo.py` | Entrena la red del cuaderno y guarda el artefacto `modelo/mnist_digits.keras` que consume la app |
| `entrenar_comparacion.py` | Entrena la densa 64-64 y la convolucional del cuaderno de clase y guarda sus artefactos en `modelo/comparacion/` |
| `tam_mnist/` | Núcleo reutilizable: preprocesamiento de imágenes (`preproceso.py`), modelo del cuaderno (`modelo.py`) y las dos redes comparadas (`comparacion.py`) |
| `tests/` | 39 tests de preprocesamiento, modelo, comparación y app, sin navegador |
| `muestras-reales/` | Seis fotos de dígitos hechos a mano con lapicera, para probar la app con fotos reales |
| `experimentos/` | Cuadernos de clase: red densa, convolucional simple y convolucional con pooling |
| `requirements.txt` | Dependencias del cuaderno y de la app |
| `requirements-dev.txt` | Dependencias para correr los tests |

**La app.** La pestaña `Subir imagen` es la entrada de la red densa 64-16 del cuaderno
`CodeTAM.ipynb`: una o varias fotos o recortes PNG/JPG del dígito hecho a mano, que la app
pasa a escala de grises, recorta y centra en 28 × 28 píxeles.

**Las tres pestañas de comparación.** El cuaderno de clase compara una red densa con una
convolucional; la app muestra esas dos redes por separado y juntas, cada una con su
predicción arriba y su anatomía abajo:

| Pestaña | Qué muestra |
| --- | --- |
| `Red densa` | La densa 64-64 (55.050 parámetros): la predicción sobre la entrada elegida y su tabla de capas |
| `Red convolucional` | La de la celda 7 del cuaderno (50.186 parámetros): predicción, tabla de capas, los 32 filtros 3 × 3 de `Convolucion_1` y los 64 mapas de activación de `Convolucion_2` para ese dígito |
| `Comparación` | Las dos predicciones sobre la misma entrada, si coinciden, cuál acierta la etiqueta real, y el contraste de parámetros y exactitud en prueba |

Cada vista pide su propia foto del dígito. Las dos redes
se entrenan con un script aparte y sus artefactos quedan en `modelo/comparacion/`, sin
tocar el artefacto de la red densa 64-16:

```bash
python entrenar_comparacion.py                 # las dos redes, 5 épocas
python entrenar_comparacion.py --redes densa   # sólo una de las dos
```

`muestras-reales/` guarda seis fotos de dígitos hechos a mano con lapicera sobre
papel y tomadas con el celular: `1`, `2`, `3`, `5`, `8` y `9`, con el dígito
fotografiado como nombre de archivo. Sirven para probar la pestaña `Subir
imagen` con fotos reales. Los tests no las usan: generan sus propias fotos
sintéticas.

`experimentos/` reúne los cuadernos de clase, en el orden en que se probaron:
`scratchpad.ipynb` (red densa de 64-64), `scratchpad_2.ipynb` (la densa y una
convolucional de dos capas) y `scratchpad_3.ipynb` (la convolucional con
`MaxPooling`, nombres en las capas y la visualización de los filtros y de las
activaciones). Se ejecutan con el mismo entorno que `CodeTAM.ipynb`.

El modelo se entrena una sola vez y queda en disco; la app sólo lo carga. Lo mismo vale
para las dos redes de comparación:

```bash
cd Talleres/Practica-MNIST
source ../../.venv312/bin/activate      # o el entorno con TensorFlow 2.21
python entrenar_modelo.py               # ~12 s: genera modelo/mnist_digits.keras
python entrenar_comparacion.py          # ~1,5 min: genera modelo/comparacion/
streamlit run app.py                    # abre la app en el navegador
```

Si `streamlit` responde «orden no encontrada», el entorno no está activado en esa shell: el
mismo comando con el intérprete del entorno es `../../.venv312/bin/python -m streamlit run app.py`.

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
| `Solución_Parcial-1_TAM_2026.pdf` | Parcial 1 resuelto, en PDF |
| `Parcial_1_TAM_2026_2_Pearson_RKHS.pdf` | Material del parcial 1 sobre el coeficiente de Pearson y RKHS, en PDF |
| `solucion_parcial_1.ipynb` | Cuaderno ejecutado de punta a punta: punto 1.5 (MCO, ridge, MV y MAP con N = 300, P = 5, Q = 20) y punto 2.4 (estimador empírico de MMD² con kernel gaussiano, aplicado solo a MNIST: primeros 500 datos contra segundos 500), con gráficas de las comparaciones |

El cuaderno usa TensorFlow solo para cargar MNIST, NumPy para las cuentas y
matplotlib para las seis figuras; no usa scikit-learn. Hay que ejecutarlo con el
entorno que tenga TensorFlow (`.venv312/`, Python 3.12) y con
`Parciales/Parcial-1/` como directorio de trabajo:

```bash
cd Parciales/Parcial-1
jupyter lab solucion_parcial_1.ipynb      # o abrirlo en VS Code y elegir el kernel
```

La primera ejecución descarga MNIST en la caché de Keras (`~/.keras/datasets/`);
después corre sin red. Un aviso de CUDA en la primera celda es normal: TensorFlow
trabaja en CPU.

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
