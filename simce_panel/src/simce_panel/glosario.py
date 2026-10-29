"""
glosario.py
===========

Generación del glosario de los archivos unificados.

Los archivos originales de la Agencia traen un glosario por año en formato
Excel (``..._GLOSAS_...xlsx``) que describe cada variable. Este módulo:

1. Lee todas las glosas disponibles y construye un diccionario
   ``{variable_normalizada: descripcion}`` (combinando años; la descripción
   más reciente y no vacía prevalece).
2. Inventaria las columnas de cada panel unificado (tipo de dato, nº de no
   nulos, % de completitud).
3. Cruza ambos para producir un glosario único de los archivos finales,
   añadiendo descripciones propias para las columnas derivadas
   (``año``, ``latitud``, ``longitud``).

El glosario se exporta a Excel con una hoja por panel más una hoja
``RESUMEN`` con el inventario de archivos.

Funciones principales
---------------------
- :func:`cargar_glosas`           : {variable_norm: descripcion} desde los xlsx.
- :func:`inventario_columnas`     : inventario de columnas de un panel.
- :func:`generar_glosario`        : glosario por panel (DataFrame).
- :func:`exportar_glosario_excel` : escribe el Excel multi-hoja.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config
from .columnas import normalizar_texto
from .io_datos import extraer_anio

log = logging.getLogger(__name__)

#: Descripciones propias para las columnas que añade el pipeline.
DESCRIPCIONES_DERIVADAS: dict[str, str] = {
    config.COL_ANIO: (
        "Año de aplicación del SIMCE (añadido por el pipeline a partir del "
        "nombre del archivo y/o la columna 'agno' original)."
    ),
    config.COL_LAT: (
        "Latitud del establecimiento (grados decimales, WGS84), anexada por "
        "cruce LEFT JOIN con el Directorio Oficial del año correspondiente."
    ),
    config.COL_LON: (
        "Longitud del establecimiento (grados decimales, WGS84), anexada por "
        "cruce LEFT JOIN con el Directorio Oficial del año correspondiente."
    ),
}


def _buscar_glosas(cfg: config.Config) -> list[tuple[int, Path]]:
    """Localiza los archivos de glosas (xlsx) dentro de las carpetas SIMCE."""
    encontrados: list[tuple[int, Path]] = []
    if not cfg.simce_dir.exists():
        return encontrados
    for xlsx in sorted(cfg.simce_dir.rglob("*.xlsx")):
        nombre = normalizar_texto(xlsx.stem)
        if "glosa" in nombre:
            anio = extraer_anio(xlsx.name) or extraer_anio(xlsx.parent.name) or 0
            encontrados.append((anio, xlsx))
    return sorted(encontrados, key=lambda t: t[0])


def _extraer_pares_glosa(xlsx: Path) -> dict[str, str]:
    """Extrae pares {variable_norm: descripcion} de un archivo de glosas.

    Las glosas suelen tener dos columnas (nombre de variable y descripción),
    pero el encabezado y la posición varían entre años. La heurística:

    - Lee todas las hojas sin asumir encabezado.
    - Para cada fila con al menos 2 celdas no vacías, toma la primera como
      nombre de variable y la segunda (o la más larga restante) como descripción.

    Devuelve ``{}`` si el archivo no se puede leer.
    """
    pares: dict[str, str] = {}
    try:
        hojas = pd.read_excel(xlsx, sheet_name=None, header=None, dtype=str)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo leer glosas %s: %s", xlsx.name, exc)
        return pares

    for _, hoja in hojas.items():
        if hoja.shape[1] < 2:
            continue
        for _, fila in hoja.iterrows():
            celdas = [str(c).strip() for c in fila.tolist()
                      if pd.notna(c) and str(c).strip() != ""]
            if len(celdas) < 2:
                continue
            variable = celdas[0]
            # La descripción es la celda restante más larga.
            descripcion = max(celdas[1:], key=len)
            clave = normalizar_texto(variable)
            # Filtra encabezados típicos ("variable", "nombre", "descripcion").
            if clave in {"variable", "nombre", "descripcion", "campo", "glosa"}:
                continue
            if clave and descripcion and clave not in pares:
                pares[clave] = descripcion
    return pares


def cargar_glosas(cfg: config.Config) -> dict[str, str]:
    """Construye el diccionario {variable_normalizada: descripcion}.

    Combina todas las glosas disponibles. Si una variable aparece en varios
    años, prevalece la primera descripción no vacía encontrada (años más
    antiguos primero por orden de :func:`_buscar_glosas`; ajustable).

    Parameters
    ----------
    cfg : config.Config

    Returns
    -------
    dict[str, str]
    """
    glosas: dict[str, str] = {}
    archivos = _buscar_glosas(cfg)
    log.info("Glosas encontradas: %d archivo(s).", len(archivos))
    for _, xlsx in archivos:
        for clave, desc in _extraer_pares_glosa(xlsx).items():
            glosas.setdefault(clave, desc)
    return glosas


def inventario_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Inventaria las columnas de un panel.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Columnas: ``variable``, ``tipo_dato``, ``no_nulos``,
        ``pct_completitud``.
    """
    n = len(df)
    filas = []
    for col in df.columns:
        no_nulos = int(df[col].notna().sum())
        filas.append({
            "variable": col,
            "tipo_dato": str(df[col].dtype),
            "no_nulos": no_nulos,
            "pct_completitud": round(100 * no_nulos / n, 2) if n else 0.0,
        })
    return pd.DataFrame(filas)


def generar_glosario(
    df: pd.DataFrame,
    glosas: dict[str, str],
) -> pd.DataFrame:
    """Genera el glosario de un panel cruzando inventario + descripciones.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel unificado.
    glosas : dict[str, str]
        Diccionario de descripciones (salida de :func:`cargar_glosas`).

    Returns
    -------
    pandas.DataFrame
        Glosario con: ``variable``, ``descripcion``, ``tipo_dato``,
        ``no_nulos``, ``pct_completitud``, ``fuente_descripcion``.
    """
    inv = inventario_columnas(df)
    descripciones = []
    fuentes = []
    for var in inv["variable"]:
        clave = normalizar_texto(var)
        if var in DESCRIPCIONES_DERIVADAS:
            descripciones.append(DESCRIPCIONES_DERIVADAS[var])
            fuentes.append("derivada (pipeline)")
        elif clave in glosas:
            descripciones.append(glosas[clave])
            fuentes.append("glosas Agencia")
        else:
            descripciones.append("")
            fuentes.append("sin descripción")
    inv.insert(1, "descripcion", descripciones)
    inv["fuente_descripcion"] = fuentes
    return inv


def exportar_glosario_excel(
    paneles: dict[str, pd.DataFrame],
    cfg: config.Config,
    glosas: dict[str, str] | None = None,
) -> Path:
    """Genera y exporta el glosario unificado en un Excel multi-hoja.

    Crea una hoja por panel (``comuna``, ``deprov``, ``region``, ``rbd``) con su
    glosario, más una hoja ``RESUMEN`` con el inventario de archivos.

    Parameters
    ----------
    paneles : dict[str, pandas.DataFrame]
        ``{nivel: panel}``.
    cfg : config.Config
    glosas : dict[str, str] | None, optional
        Diccionario de descripciones. Si es ``None``, se calcula con
        :func:`cargar_glosas`.

    Returns
    -------
    Path
        Ruta del archivo Excel generado.
    """
    if glosas is None:
        glosas = cargar_glosas(cfg)

    cfg.crear_salida()
    ruta = cfg.ruta_salida(config.NOMBRE_GLOSARIO)

    resumen_filas = []
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        for nivel, df in paneles.items():
            if df.empty:
                continue
            glos = generar_glosario(df, glosas)
            hoja = nivel[:31]  # Excel limita a 31 caracteres el nombre de hoja.
            glos.to_excel(writer, sheet_name=hoja, index=False)
            resumen_filas.append({
                "archivo": config.NOMBRES_SALIDA.get(nivel, nivel),
                "nivel": nivel,
                "filas": len(df),
                "columnas": df.shape[1],
                "con_descripcion": int((glos["fuente_descripcion"]
                                        != "sin descripción").sum()),
            })
        resumen = pd.DataFrame(resumen_filas)
        if not resumen.empty:
            resumen.to_excel(writer, sheet_name="RESUMEN", index=False)

    log.info("Glosario unificado exportado: %s", ruta)
    return ruta
