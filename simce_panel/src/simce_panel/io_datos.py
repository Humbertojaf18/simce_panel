"""
io_datos.py
===========

Entrada/salida de datos y utilidades de bajo nivel para los archivos del
proyecto (CSV con separador ``;`` y codificación ``utf-8-sig``).

Funciones principales
---------------------
- :func:`leer_csv`          : lee un CSV con los parámetros del proyecto.
- :func:`escribir_csv`      : escribe un CSV con los parámetros del proyecto.
- :func:`extraer_anio`      : extrae el año (4 dígitos) de un nombre/ruta.
- :func:`a_numerico`        : convierte una serie a numérico tolerando coma decimal.
- :func:`listar_subcarpetas`: lista subcarpetas que contienen un año.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Año plausible para datos educativos chilenos (evita capturar números sueltos).
_PATRON_ANIO = re.compile(r"(20[0-4]\d)")


def extraer_anio(texto: str | Path) -> int | None:
    """Extrae el primer año de 4 dígitos (2000-2049) de un texto o ruta.

    Sirve tanto para nombres de archivo (``simce2m2015_comuna_...``) como para
    nombres de carpeta (``Simce 2° Medio 2015``) o de Directorio
    (``Directorio_oficial_EE_2014_estandarizado``).

    Parameters
    ----------
    texto : str | Path
        Cadena o ruta de la que extraer el año.

    Returns
    -------
    int | None
        El año como entero, o ``None`` si no se encuentra.

    Examples
    --------
    >>> extraer_anio("simce2m2015_comuna_publica_final.csv")
    2015
    >>> extraer_anio("Directorio_oficial_EE_2014_estandarizado.csv")
    2014
    """
    m = _PATRON_ANIO.search(str(texto))
    return int(m.group(1)) if m else None


#: Codificaciones que se prueban, en orden, cuando la indicada falla.
#: Los archivos del MINEDUC y de la Agencia no son homogéneos: la mayoría
#: viene en UTF-8 con BOM, pero algunos años están guardados en Latin-1 /
#: Windows-1252 (típico si el archivo pasó por Excel en Windows).
ENCODINGS_RESPALDO: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def leer_csv(
    ruta: str | Path,
    *,
    sep: str = ";",
    encoding: str = "utf-8-sig",
    **kwargs,
) -> pd.DataFrame:
    """Lee un CSV con los parámetros estándar del proyecto.

    Usa ``dtype=str`` por defecto NO; en su lugar deja que pandas infiera,
    pero con ``low_memory=False`` para evitar advertencias de tipos mixtos
    en archivos grandes (típico en datasets longitudinales).

    Parameters
    ----------
    ruta : str | Path
        Ruta del archivo CSV.
    sep : str, optional
        Separador de campos (por defecto ``;``).
    encoding : str, optional
        Codificación (por defecto ``utf-8-sig``).
    **kwargs
        Argumentos adicionales para :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    kwargs.setdefault("low_memory", False)

    # Se intenta primero la codificación pedida y luego las de respaldo.
    # 'latin-1' nunca falla al decodificar, por lo que actúa de última red:
    # es preferible leer con posibles acentos imperfectos que abortar el
    # pipeline completo por un archivo mal guardado.
    intentos: list[str] = [encoding] + [e for e in ENCODINGS_RESPALDO
                                        if e != encoding]
    ultimo_error: Exception | None = None
    for enc in intentos:
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, **kwargs)
            if enc != encoding:
                log.warning("%s no se pudo leer como '%s'; se leyó como '%s'.",
                            Path(ruta).name, encoding, enc)
            return df
        except UnicodeDecodeError as exc:
            ultimo_error = exc
            continue
    raise UnicodeDecodeError(
        "utf-8", b"", 0, 1,
        f"No se pudo decodificar {Path(ruta).name} con ninguna de estas "
        f"codificaciones: {', '.join(intentos)}. Detalle: {ultimo_error}",
    )


def escribir_csv(
    df: pd.DataFrame,
    ruta: str | Path,
    *,
    sep: str = ";",
    encoding: str = "utf-8-sig",
    index: bool = False,
    **kwargs,
) -> Path:
    """Escribe un DataFrame a CSV con los parámetros estándar del proyecto.

    Crea la carpeta de destino si no existe.

    Parameters
    ----------
    df : pandas.DataFrame
    ruta : str | Path
    sep : str, optional
    encoding : str, optional
    index : bool, optional
        Si se escribe el índice (por defecto ``False``).

    Returns
    -------
    Path
        La ruta del archivo escrito.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, sep=sep, encoding=encoding, index=index, **kwargs)
    return ruta


def a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte una serie a numérico tolerando coma decimal y espacios.

    Pensada para coordenadas y puntajes que a veces vienen como texto con
    coma decimal (``"-33,45"``). Los valores no convertibles quedan como NaN.

    Parameters
    ----------
    serie : pandas.Series

    Returns
    -------
    pandas.Series
        Serie de tipo float con NaN donde no se pudo convertir.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype("float64")
    s = (
        serie.astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    s = s.replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(s, errors="coerce")


def listar_subcarpetas_con_anio(carpeta: str | Path) -> list[tuple[int, Path]]:
    """Lista subcarpetas cuyo nombre contiene un año, ordenadas por año.

    Parameters
    ----------
    carpeta : str | Path
        Carpeta padre (por ejemplo, ``Simce 2°medio``).

    Returns
    -------
    list[tuple[int, Path]]
        Lista de ``(anio, ruta_subcarpeta)`` ordenada ascendentemente por año.
    """
    carpeta = Path(carpeta)
    encontrados: list[tuple[int, Path]] = []
    for sub in sorted(carpeta.iterdir()):
        if not sub.is_dir():
            continue
        anio = extraer_anio(sub.name)
        if anio is not None:
            encontrados.append((anio, sub))
    return sorted(encontrados, key=lambda t: t[0])
