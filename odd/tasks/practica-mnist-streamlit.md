# Practica-MNIST como app Streamlit

Convertir la práctica de dígitos MNIST (hoy un cuaderno) en una app Streamlit que
cumple la tarea: cargar una imagen de un dígito escrito a mano y que el modelo del
cuaderno prediga cuál es. El cuaderno `CodeTAM.ipynb` se conserva intacto.

## Specs

**S1** — Pedido del usuario, literal: `convierte la "PRACTICA MNIST" en un streamlit`

**S2** — Enunciado de la tarea, literal de `Practica-MNIST/TAREA-TAM.txt`:
`Crea una interfaz  en donde se pueda cargar un numero escrito a mano(imagen) y a partir del modelo plasmado del cuaderno.ipynb  predizca el  número qué es`

**S3** — Formas de entrada (decisión del usuario): las tres deben existir.
- Subir una o varias imágenes PNG/JPG.
- Lienzo para dibujar un dígito con el mouse.
- Elegir una muestra del conjunto de prueba MNIST.

**S4** — Entrenamiento (decisión del usuario): un script `entrenar_modelo.py`
entrena y guarda el artefacto `.keras`; la app sólo lo carga. Sin entrenamiento en
el arranque de la app.

**S5** — Paridad con el cuaderno `CodeTAM.ipynb`:
- Arquitectura: entrada `784` → `Dense 64 relu` → `Dense 16 relu` → `Dense 10 softmax`.
- Compilación: Adam, `SparseCategoricalCrossentropy(from_logits=False)`, métrica
  `SparseCategoricalAccuracy`.
- Entrenamiento: `epochs=5`, `batch_size=128`, `validation_split=0.1`, `np.random.seed(42)`, `tf.random.set_seed(42)`.
- Preprocesamiento de una imagen externa: escala de grises, autocontraste, invertir
  si el fondo es claro, recortar el trazo, escalar a 20×20 como máximo y centrar en
  un lienzo 28×28 negro con **el centro de masa del trazo** en el centro (ver L5).
- Mensaje de error, literal: `No se encontró un dígito visible en la imagen.`

**S6** — La app muestra la exactitud del modelo sobre el conjunto de prueba y la
tabla de arquitectura (entrada + 3 capas `Dense`, parámetros entrenables).

## Tasks

| ID | Specs | Qué | Ruta | Commit |
| --- | --- | --- | --- | --- |
| T1 | S4 | Dependencias Streamlit/canvas/pytest en `requirements.txt` y venv 3.12 | inline | sí |
| T2 | S5 | Paquete `tam_mnist`: preprocesamiento de imágenes y lienzo, con tests | inline | sí |
| T3 | S5 | Paquete `tam_mnist`: construcción/entrenamiento/persistencia del modelo, con tests | inline | sí |
| T4 | S3, S6 | `app.py`: las tres entradas, predicción, métricas y tabla de arquitectura | inline | sí |
| T5 | S4 | `entrenar_modelo.py` ejecutado, artefacto generado, app verificada headless | inline | sí |
| T6 | — | README y `.gitignore`; el cuaderno queda como estaba | inline | sí |
| T7 | S5 | Centrado por centro de masa en vez de por caja, con tests | inline | sí |

## Log

**L1** — `convierte la "PRACTICA MNIST" en un streamlit`

**L2** — `vuelve a preguntarme cuando se entrena la red, dame nuevamente la opción multiple`

**L3** — Decisiones de las preguntas anteriores: entradas = subir imagen + lienzo + test set (las tres); entrenamiento = script + artefacto `.keras` cacheado.

**L4** — Contexto de entorno ya resuelto antes de esta feature: Python 3.14 del sistema no tiene wheels de TensorFlow 2.21, así que existe `.venv312/` (CPython 3.12.16, TensorFlow 2.21.0) documentado en `README.md`. La app corre en ese venv.

**L5** — Evidencia medida antes de cerrar el preprocesamiento. Se convirtieron 100 imágenes del test set a fotos sintéticas (dígito escalado a 88×88 pegado en una hoja clara de 200×200) y se pasaron por el camino real de subida:

| Preprocesamiento | Aciertos |
| --- | --- |
| Centrado geométrico (celda 13 del cuaderno) | 71/100 |
| Centrado por centro de masa | 98/100 |
| MNIST original, sin preprocesar | 98/100 |

Causa: el cuaderno centra por el punto medio de la caja del trazo, mientras que MNIST normaliza por centro de masa. El corrimiento de 2–4 píxeles basta para que la red densa —que no es invariante a traslaciones— confunda un 7 con un 2. Decisión del usuario: `Centro de masa y listo`. La arquitectura, los pesos y el entrenamiento quedan idénticos a los del cuaderno; cambia sólo el centrado de las imágenes externas. Medición final por `preprocess_bytes`: 95/100.

**L6** — Verificación del artefacto entrenado: exactitud 96.01 % en prueba (5 épocas, semilla 42), `AppTest` contra el modelo real sin excepciones, primera imagen del test set predicha como 7 con 99.5 % (etiqueta real 7), 19/20 aciertos en las primeras 20 imágenes, y servidor `streamlit run app.py` con `/_stcore/health` en `ok` y la página en 200.

## Slices

El repo entrega por unidades de trabajo revisables; cada commit es una rebanada
que se sostiene sola.

| Slice | Commit | Contenido |
| --- | --- | --- |
| 1 | `2cf8210` | `tam_mnist/preproceso.py` y sus tests (421 líneas) |
| 2 | `59b673a` | `tam_mnist/modelo.py`, `entrenar_modelo.py` y sus tests |
| 3 | `56275af` | `app.py`, sus tests y las dependencias |
| 4 | `1a24cc1` | README, `.gitignore` y este documento |
| 5 | `f988418` | Reorganización del repositorio en `Talleres/` y `Parciales/`, README reescrito |

## Log posterior (fotos reales y reorganización)

**L7** — El usuario reportó que la app fallaba con una foto de un dígito hecho a mano sobre papel con lapicero negro. Reproducción con un banco de fotos sintéticas (900×900, ruido de sensor, sombra lateral, escritorio en el borde, JPEG, subexposición; 30 dígitos por escenario): con el preproceso de umbral fijo, ruido y sombra daban 0/30, escritorio 1/30 y hoja en blanco ni siquiera se rechazaba. Diagnóstico: el umbral fijo sobre la imagen completa detectaba ruido y sombra como tinta, el "trazo" pasaba a ser la foto entera y el dígito se achicaba hasta desaparecer.

Arreglo: fondo local con desenfoque, separación de tinta con polaridad elegida por masa, umbral de Otsu con piso absoluto (`MIN_INK`), selección del conjunto de tinta más pesado sin tocar el borde, relleno de huecos y centrado por masa. Sus reglas quedaron pinneadas en `tests/test_preproceso.py` (25 tests verdes en total, sin modelo ni dataset).

**L8** — Verificación con las fotos reales del usuario (`~/Descargas/1.jpeg`, `3.jpeg`, `5.jpeg`, `9.jpeg`, fotos de celular): 1→1 con 98.9 %, 3→3 con 99.8 %, 5→5 con 99.1 %, 9→9 con 96.6 %. El fallo original que reportó el usuario venía de la versión anterior del preproceso, que era la que tenía cargada la app en ese momento.

Queda un hueco conocido: en el banco sintético los escenarios de ruido, sombra y escritorio siguen bajos, mientras las fotos reales funcionan. El generador sintético resultó más duro que la realidad; se prioriza el comportamiento real verificado sobre el ajuste al generador.

**L9** — El usuario reorganizó el repositorio (`Talleres/Practica-MNIST`, `Talleres/Taller-ALternativo-Repaso`, `Parciales/Parcial-1`) y pidió publicar. `TAREA-TAM.txt` había quedado fuera de la mudanza y se restauró desde Git en su ruta nueva. El `.gitignore` del artefacto se movió con la carpeta.
