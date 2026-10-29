"""Genera la muestra pública de Dataset/muestra a partir del panel reducido completo.

Uso (desde la raíz del repositorio):
    python herramientas/generar_muestra.py \
        Dataset/Unificados/simce_rbd_ancho_reducido_2015_2025.csv \
        Dataset/Unificados/simce_rbd_unificado_2015_2025.csv

Toma el 15 % de los establecimientos, estratificado por región (semilla 2026),
recalcula con simce_panel las variables de resultado de Matemática y Lectura
para esa muestra, con las mismas reglas que el cuaderno 02 (min_pre=2, min_post=3),
y extrae del panel largo las columnas que usan los cuadernos 04, 06 y 09
(dependencia y GSE por año, para los traspasos a los SLEP y el GSE previo).

El panel de entrada debe venir saneado (cuaderno 03 con simce_panel 1.1.2 o
posterior): sin los puntajes que la Agencia marca como no representativos. Por
si acaso, aquí se vuelven a anular los códigos de supresión (< 100 puntos).
Por la medida ética del informe (apartado 5.4), el nombre del establecimiento se
reemplaza por un rótulo genérico; el RBD y las coordenadas se conservan porque
los cuadernos los necesitan y ya son públicos en el Directorio Oficial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "simce_panel" / "src"))
import simce_panel as sp  # noqa: E402
from simce_panel import resultado as R  # noqa: E402

FRACCION, SEMILLA = 0.15, 2026
ANIOS = (2015, 2016, 2017, 2018, 2022, 2023, 2024, 2025)


def a_largo(ancho: pd.DataFrame) -> pd.DataFrame:
    """Pasa el panel ancho reducido a formato largo (una fila por RBD y año)."""
    partes = []
    for t in ANIOS:
        p = ancho[["rbd", "latitud", "longitud"]].copy()
        p["año"] = t
        for m in ("mate", "lect"):
            p[f"prom_{m}2m_rbd"] = pd.to_numeric(ancho.get(f"prom_{m}2m_rbd_{t}"), errors="coerce")
            p[f"nalu_{m}2m_rbd"] = pd.to_numeric(ancho.get(f"nalu_{m}2m_rbd_{t}"), errors="coerce")
        partes.append(p)
    return pd.concat(partes, ignore_index=True)


COLS_LARGO = ["rbd", "año", "nom_rbd", "cod_depe1", "cod_depe2", "cod_grupo",
              "cod_rural_rbd", "prom_lect2m_rbd", "prom_mate2m_rbd",
              "nalu_lect2m_rbd", "nalu_mate2m_rbd", "marca_lect2m_rbd",
              "marca_mate2m_rbd", "latitud", "longitud"]


def main(ruta_ancho: str, ruta_largo: str | None = None,
         salida: str = "Dataset/muestra") -> None:
    ancho = sp.io_datos.leer_csv(ruta_ancho)
    muestra = (ancho.groupby("region_dir", group_keys=False, dropna=False)
                    .sample(frac=FRACCION, random_state=SEMILLA)
                    .sort_values("rbd").reset_index(drop=True))
    for c in [c for c in muestra.columns if c.startswith(("prom_mate", "prom_lect"))]:
        v = pd.to_numeric(muestra[c], errors="coerce")
        muestra[c] = v.where(v >= 100)
    if "nom_rbd" in muestra.columns:
        muestra["nom_rbd"] = "Establecimiento " + muestra["rbd"].astype(str)
    largo = a_largo(muestra)
    res_m = R.construir_resultados(largo, "prom_mate2m_rbd", min_pre=2, min_post=3)
    res_m = R.clasificar_cambio_nivel(largo, res_m, "prom_mate2m_rbd")
    res_l = R.construir_resultados(largo, "prom_lect2m_rbd", min_pre=2, min_post=3)
    out = Path(salida); out.mkdir(parents=True, exist_ok=True)
    sp.io_datos.escribir_csv(muestra, out / "simce_rbd_ancho_reducido_2015_2025.csv")
    sp.io_datos.escribir_csv(res_m, out / "resultados_mate_2m.csv")
    sp.io_datos.escribir_csv(res_l, out / "resultados_lect_2m.csv")
    if ruta_largo:
        largo_full = sp.io_datos.leer_csv(ruta_largo)
        largo_m = largo_full[largo_full["rbd"].isin(set(muestra["rbd"]))]
        largo_m = largo_m[[c for c in COLS_LARGO if c in largo_m.columns]].copy()
        largo_m, _ = sp.diagnostico.sanear_puntajes(
            largo_m, materias=("lect", "mate"), min_valido=100, max_valido=None)
        if "nom_rbd" in largo_m.columns:
            largo_m["nom_rbd"] = "Establecimiento " + largo_m["rbd"].astype(str)
        sp.io_datos.escribir_csv(largo_m, out / "simce_rbd_unificado_2015_2025.csv")
    print(f"Muestra: {len(muestra)} de {len(ancho)} establecimientos | "
          f"analítica Mate {int(res_m.muestra_analitica.sum())}, "
          f"Lect {int(res_l.muestra_analitica.sum())}")


if __name__ == "__main__":
    main(*sys.argv[1:])
