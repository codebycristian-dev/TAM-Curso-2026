"""Ejecuta `solucion_taller.ipynb` de punta a punta y reporta errores.

Uso:  python scripts/ejecutar_notebook.py
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent.parent
RUTA = RAIZ / "solucion_taller.ipynb"

nb = nbformat.read(RUTA, as_version=4)
cliente = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(RAIZ)}},
)
cliente.execute()
nbformat.write(nb, RUTA)

salidas = []
for i, celda in enumerate(nb.cells):
    if celda.cell_type != "code":
        continue
    for out in celda.get("outputs", []):
        if out.get("output_type") == "stream":
            salidas.append(out["text"])
        elif out.get("output_type") == "error":
            salidas.append(f"ERROR en celda {i}:\n" + "\n".join(out["traceback"]))
print("".join(salidas))
print(f"\nOK: cuaderno ejecutado y guardado en {RUTA}")
