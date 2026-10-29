"""
simce_panel.matricula_curso
===========================

Matrícula exacta de 2.º medio por establecimiento y tasa de participación en el
SIMCE, a partir del Resumen de Matrícula por Curso del MINEDUC
(datosabiertos.mineduc.cl, un CSV por año con corte al 30 de abril).

Reemplaza a la aproximación «un cuarto de la matrícula de media» del cuaderno 07.

Reglas, según el esquema de registro (ER_Matricula_por_Curso.pdf):
- Enseñanza media de niños y jóvenes: COD_ENSE 310 (HC), 410, 510, 610, 710,
  810 (TP) y 910 (artística). Se excluyen adultos (363, 463, …) y especial.
- Cursos simples (COD_TIP_CUR = 0): el grado está en COD_GRADO y los alumnos en
  N_ALU. Cursos combinados (COD_TIP_CUR > 0): los alumnos de cada grado están en
  N_ALU_GRADO1 … N_ALU_GRADO9. En media, 2015–2025 no hay cursos combinados,
  pero el código los cubre igual.
- Solo establecimientos funcionando (ESTADO_ESTAB = 1) y, desde 2018, se
  excluyen las aulas hospitalarias (TIPO_AULA = 2).
- Los archivos cambian entre años: nombres de columna en minúsculas hasta 2017,
  codificación local hasta 2017 y UTF-8 con BOM después, columnas nuevas desde
  2022. La lectura normaliza todo eso.

Uso en el cuaderno 07:

    from simce_panel.matricula_curso import matricula_2m, participacion_exacta
    mat2m = matricula_2m(EXT)                    # EXT = Dataset/Externos
    part = participacion_exacta(ancho, mat2m)    # rbd, anio, evaluados, mat_2m, tasa_participacion
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("simce_panel.matricula_curso")

MEDIA_JOVENES = (310, 410, 510, 610, 710, 810, 910)
ANIOS = (2015, 2016, 2017, 2018, 2022, 2023, 2024, 2025)
_NUMERICAS = ["AGNO", "RBD", "COD_ENSE", "COD_GRADO", "COD_TIP_CUR", "ESTADO_ESTAB", "TIPO_AULA", "N_ALU",
              "N_ALU_GRADO1", "N_ALU_GRADO2", "N_ALU_GRADO3", "N_ALU_GRADO4"]


def leer_resumen_curso(ruta: str | Path) -> pd.DataFrame:
    """Lee un CSV del Resumen de Matrícula por Curso de cualquier año (2015–2025)."""
    ultimo_error = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(ruta, sep=";", encoding=enc, dtype=str, low_memory=False)
            break
        except UnicodeDecodeError as e:
            ultimo_error = e
    else:
        raise ultimo_error
    d.columns = [c.strip().upper() for c in d.columns]
    for c in _NUMERICAS:
        if c in d:
            d[c] = pd.to_numeric(d[c].str.strip().replace("", np.nan), errors="coerce")
    return d


def matricula_media_por_rbd(d: pd.DataFrame) -> pd.DataFrame:
    """Matrícula de 1.º a 4.º medio de jóvenes por RBD y la de 2.º medio por separado."""
    m = d[d["COD_ENSE"].isin(MEDIA_JOVENES) & (d["ESTADO_ESTAB"] == 1)].copy()
    if "TIPO_AULA" in m:
        m = m[m["TIPO_AULA"] != 2]
    simple = m["COD_TIP_CUR"].fillna(0) == 0
    for g in (1, 2, 3, 4):
        m[f"g{g}"] = (np.where(simple & (m["COD_GRADO"] == g), m["N_ALU"].fillna(0), 0)
                      + np.where(~simple, m[f"N_ALU_GRADO{g}"].fillna(0), 0))
    r = m.groupby("RBD")[["g1", "g2", "g3", "g4"]].sum()
    out = pd.DataFrame({"rbd": r.index.astype(int), "mat_2m": r["g2"].to_numpy(),
                        "mat_media": r[["g1", "g2", "g3", "g4"]].sum(axis=1).to_numpy()})
    return out


def matricula_2m(ext_dir: str | Path, anios=ANIOS) -> pd.DataFrame:
    """Busca un CSV por año dentro de ext_dir (en subcarpetas) y devuelve rbd, anio, mat_2m, mat_media."""
    ext_dir = Path(ext_dir)
    archivos = {}
    for f in ext_dir.rglob("*.csv"):
        m = re.search(r"Curso_(20\d{2})", f.name, flags=re.IGNORECASE)
        if m and "frecuencia" not in f.name.lower():
            archivos[int(m.group(1))] = f
    faltan = [a for a in anios if a not in archivos]
    if faltan:
        raise FileNotFoundError(f"No encontré el Resumen de Matrícula por Curso de: {faltan}")
    partes = []
    for a in anios:
        r = matricula_media_por_rbd(leer_resumen_curso(archivos[a]))
        r["anio"] = a
        log.info("Matrícula por curso %s: %d establecimientos con media de jóvenes; 2.º medio = %d",
                 a, len(r), int(r["mat_2m"].sum()))
        partes.append(r)
    return pd.concat(partes, ignore_index=True)[["rbd", "anio", "mat_2m", "mat_media"]]


def participacion_exacta(ancho: pd.DataFrame, mat2m: pd.DataFrame, asignatura: str = "mate",
                         anios=ANIOS) -> pd.DataFrame:
    """Tasa de participación = evaluados en 2.º medio / matrícula de 2.º medio (por RBD y año).

    `ancho` es el panel ancho del cuaderno 03 con las columnas nalu_<asignatura>2m_rbd_<año>.
    Devuelve las mismas columnas que usaba la versión aproximada, más mat_2m.
    """
    filas = []
    for a in anios:
        col = f"nalu_{asignatura}2m_rbd_{a}"
        t = ancho[["rbd", col]].rename(columns={col: "evaluados"}).dropna()
        t["anio"] = a
        filas.append(t)
    ev = pd.concat(filas, ignore_index=True)
    part = ev.merge(mat2m[["rbd", "anio", "mat_2m"]], on=["rbd", "anio"], how="inner")
    part = part[part["mat_2m"] > 0].copy()
    part["tasa_participacion"] = part["evaluados"] / part["mat_2m"]
    n_alta = int((part["tasa_participacion"] > 1.05).sum())
    log.info("Tasa exacta: %d registros; %d (%.1f %%) sobre 1,05", len(part), n_alta, 100 * n_alta / len(part))
    return part[["rbd", "anio", "evaluados", "mat_2m", "tasa_participacion"]]
