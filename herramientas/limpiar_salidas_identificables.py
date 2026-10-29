"""Quita de los cuadernos ejecutados las salidas que identifican establecimientos.

Uso (desde la raíz del repositorio, antes de publicar):
    python herramientas/limpiar_salidas_identificables.py            # todos los .ipynb de la raíz
    python herramientas/limpiar_salidas_identificables.py 02_revision_calidad.ipynb

Coherente con la medida ética del informe (apartado 5.4): las salidas que muestran
nombres de establecimientos se reemplazan por un aviso; el resto de las salidas
(cifras, tablas agregadas, figuras) se conserva. También reemplaza las rutas locales
de Windows por <ruta local>. Modifica los archivos en su lugar: córrelo sobre la
copia que vas a publicar, no sobre tu carpeta de trabajo.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATRON_NOMBRE = re.compile(
    r"\bnom_rbd\b|\b(COLEGIO|LICEO|ESCUELA|ESC\.|INSTITUTO|COMPLEJO EDUCACIONAL|"
    r"CENTRO EDUCACIONAL|COLLEGE|SCHOOL|SAINT|ACADEMIA)\b")
PATRON_RUTA = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s\"'|<>]+\\)*[^\\\s\"'|<>]*")
AVISO = {"name": "stdout", "output_type": "stream",
         "text": ["[Salida omitida en la versión pública: mostraba nombres de establecimientos.]\n"]}


def _texto(salida: dict) -> str:
    return json.dumps(salida, ensure_ascii=False)


def _sin_rutas(obj):
    """Reemplaza rutas locales en todas las cadenas de una salida."""
    if isinstance(obj, str):
        nuevo, n = PATRON_RUTA.subn("<ruta local>", obj)
        return nuevo, n
    if isinstance(obj, list):
        pares = [_sin_rutas(x) for x in obj]
        return [x for x, _ in pares], sum(n for _, n in pares)
    if isinstance(obj, dict):
        pares = {k: _sin_rutas(v) for k, v in obj.items()}
        return {k: v for k, (v, _) in pares.items()}, sum(n for _, n in pares.values())
    return obj, 0


def limpiar(ruta: Path) -> tuple[int, int]:
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    omitidas = rutas = 0
    for celda in nb.get("cells", []):
        if celda.get("cell_type") != "code":
            continue
        nuevas, hubo = [], False
        for salida in celda.get("outputs", []):
            if PATRON_NOMBRE.search(_texto(salida)):
                omitidas += 1
                hubo = True
                continue
            salida, n = _sin_rutas(salida)
            rutas += n
            nuevas.append(salida)
        if hubo:
            nuevas.append(AVISO)
        celda["outputs"] = nuevas
    ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return omitidas, rutas


def main(args: list[str]) -> None:
    archivos = [Path(a) for a in args] or sorted(Path(".").glob("*.ipynb"))
    if not archivos:
        print("No encontré cuadernos .ipynb en esta carpeta.")
        return
    for ruta in archivos:
        omitidas, rutas = limpiar(ruta)
        print(f"{ruta.name}: {omitidas} salidas omitidas, {rutas} rutas locales reemplazadas")


if __name__ == "__main__":
    main(sys.argv[1:])
