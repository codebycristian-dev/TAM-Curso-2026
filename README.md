# TAM — Curso 2026

Materiales de **Teoría de Aprendizaje de Máquina (2026-2)**: la práctica de
clasificación de dígitos escritos a mano con MNIST y el taller de repaso con su
informe.

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `Practica-MNIST/` | Cuaderno de la práctica, enunciado de la tarea y sus dependencias |
| `Taller-Repaso/` | Enunciado, solución, figuras, informe y scripts del taller |

### `Practica-MNIST/`

| Archivo | Qué es |
| --- | --- |
| `CodeTAM.ipynb` | Prepara MNIST, entrena una red densa, la evalúa y permite probar una imagen propia (en Colab y en VS Code) |
| `TAREA-TAM.txt` | Enunciado de la tarea: interfaz para cargar un dígito escrito a mano y predecirlo con el modelo del cuaderno |
| `requirements.txt` | Dependencias de Python del cuaderno |

Ejecutar el cuaderno:

```bash
cd Practica-MNIST
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Después abrir `CodeTAM.ipynb` y ejecutar las celdas en orden. El directorio de
trabajo debe ser `Practica-MNIST/`, porque el cuaderno instala desde
`requirements.txt` y resuelve rutas relativas a esa carpeta. La primera
ejecución descarga MNIST y necesita conexión a Internet.

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
