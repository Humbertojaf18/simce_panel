"""
validacion.py
=============

Funciones de validación y diagnóstico para los paneles unificados.

Permiten confirmar que la unificación se hizo bien: número de filas por año,
vista previa de las primeras filas y verificación de que la columna ``año``
quedó correctamente poblada.

Funciones principales
---------------------
- :func:`resumen_por_anio`     : filas por año (y, opcionalmente, por nivel).
- :func:`verificar_columna_anio`: chequeos de integridad de la columna ``año``.
- :func:`preview`              : primeras filas de un DataFrame.
- :func:`reporte_unificacion`  : reporte consolidado de los 4 paneles.
"""

from __future__ import annotations

import pandas as pd

from . import config


def resumen_por_anio(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta el número de filas por año en un panel.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel con la columna :data:`config.COL_ANIO`.

    Returns
    -------
    pandas.DataFrame
        Tabla con columnas ``año`` y ``filas``.
    """
    if config.COL_ANIO not in df.columns or df.empty:
        return pd.DataFrame(columns=[config.COL_ANIO, "filas"])
    out = (
        df.groupby(config.COL_ANIO, dropna=False)
        .size()
        .reset_index(name="filas")
        .sort_values(config.COL_ANIO)
        .reset_index(drop=True)
    )
    return out


def verificar_columna_anio(df: pd.DataFrame) -> dict[str, object]:
    """Verifica la integridad de la columna ``año``.

    Comprueba que la columna exista, no tenga nulos y que los años estén dentro
    de un rango plausible.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Diccionario con: ``existe``, ``nulos``, ``anios``, ``ok``.
    """
    existe = config.COL_ANIO in df.columns
    if not existe:
        return {"existe": False, "nulos": None, "anios": [], "ok": False}
    serie = df[config.COL_ANIO]
    nulos = int(serie.isna().sum())
    anios = sorted(serie.dropna().unique().tolist())
    rango_ok = all(2000 <= int(a) <= 2049 for a in anios) if anios else False
    ok = (nulos == 0) and rango_ok and len(anios) > 0
    return {"existe": existe, "nulos": nulos, "anios": anios, "ok": ok}


def preview(df: pd.DataFrame, n: int = 3, max_cols: int = 12) -> pd.DataFrame:
    """Devuelve las primeras ``n`` filas, limitando columnas para legibilidad.

    Parameters
    ----------
    df : pandas.DataFrame
    n : int, optional
        Número de filas (por defecto 3).
    max_cols : int, optional
        Máximo de columnas a mostrar (por defecto 12).

    Returns
    -------
    pandas.DataFrame
    """
    if df.empty:
        return df
    cols = list(df.columns[:max_cols])
    return df.head(n)[cols]


def reporte_unificacion(paneles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Genera un reporte consolidado de los paneles unificados.

    Parameters
    ----------
    paneles : dict[str, pandas.DataFrame]
        ``{nivel: panel}``.

    Returns
    -------
    pandas.DataFrame
        Una fila por nivel con: filas, columnas, n.º de años, años, año OK.
    """
    filas = []
    for nivel, df in paneles.items():
        chk = verificar_columna_anio(df)
        filas.append({
            "nivel": nivel,
            "filas": len(df),
            "columnas": df.shape[1],
            "n_anios": len(chk["anios"]),
            "anios": ", ".join(str(a) for a in chk["anios"]),
            "anio_ok": chk["ok"],
        })
    return pd.DataFrame(filas)
