"""
columnas.py
===========

Utilidades para trabajar con nombres de columnas que cambian entre años.

Los archivos del MINEDUC y de la Agencia de Calidad no son perfectamente
consistentes: una misma variable puede aparecer como ``rbd``, ``RBD`` o
``cod_rbd``; las coordenadas como ``LATITUD``, ``GPS_LAT`` o ``latitud``.
Este módulo permite localizar la columna real a partir de una lista de
*alias*, comparándolos de forma robusta (sin acentos, en minúsculas y sin
separadores).

Funciones principales
---------------------
- :func:`normalizar_texto`      : normaliza un string para comparación.
- :func:`resolver_columna`      : busca una columna por alias (o ``None``).
- :func:`resolver_columna_req`  : igual, pero lanza error si no la encuentra.
- :func:`mapa_normalizado`      : dict {nombre_normalizado: nombre_real}.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import pandas as pd


def normalizar_texto(texto: str) -> str:
    """Normaliza un texto para comparar nombres de columnas de forma robusta.

    Pasa a minúsculas, elimina acentos/diacríticos y quita todo lo que no sea
    alfanumérico (espacios, guiones, guiones bajos, etc.).

    Parameters
    ----------
    texto : str
        Texto original (por ejemplo, un nombre de columna).

    Returns
    -------
    str
        Texto normalizado. ``"GPS_LAT"`` y ``"gps lat"`` -> ``"gpslat"``.

    Examples
    --------
    >>> normalizar_texto("LATITUD")
    'latitud'
    >>> normalizar_texto("Año")
    'ano'
    >>> normalizar_texto("cod_RBD ")
    'codrbd'
    """
    if texto is None:
        return ""
    texto = str(texto)
    # Descompone los acentos y los elimina (NFKD).
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    # Conserva solo caracteres alfanuméricos.
    return re.sub(r"[^a-z0-9]", "", texto)


def mapa_normalizado(df: pd.DataFrame) -> dict[str, str]:
    """Devuelve un diccionario {nombre_normalizado: nombre_real}.

    Útil para resolver columnas. Si dos columnas colapsan al mismo nombre
    normalizado, se conserva la primera (orden de aparición).

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict[str, str]
    """
    mapa: dict[str, str] = {}
    for col in df.columns:
        clave = normalizar_texto(col)
        mapa.setdefault(clave, col)
    return mapa


def unificar_nombres_columnas(
    bloques: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Homologa los nombres de columna equivalentes entre varios DataFrames.

    Problema que resuelve: si un año trae la columna ``RBD`` y otro ``rbd``
    (o ``Prom_Lect`` vs ``prom_lect``), al concatenar pandas las trata como
    columnas distintas y los datos quedan partidos en dos columnas con muchos
    NaN. Esta función detecta columnas que normalizan al mismo nombre y las
    renombra a una grafía canónica común (la más frecuente entre los bloques),
    de modo que tras la concatenación haya UNA sola columna por concepto.

    Parameters
    ----------
    bloques : list[pandas.DataFrame]
        DataFrames a homologar (por ejemplo, un panel por año).

    Returns
    -------
    list[pandas.DataFrame]
        Los mismos DataFrames con las columnas renombradas a la grafía canónica.
        (Se devuelven copias con columnas renombradas; los datos no se copian.)
    """
    from collections import Counter

    # Cuenta la grafía original de cada nombre normalizado en todos los bloques.
    conteo: dict[str, Counter] = {}
    for df in bloques:
        for col in df.columns:
            clave = normalizar_texto(col)
            conteo.setdefault(clave, Counter())[col] += 1

    # Grafía canónica = la más frecuente (desempate: la primera vista).
    canonico: dict[str, str] = {
        clave: contador.most_common(1)[0][0]
        for clave, contador in conteo.items()
    }

    renombrados: list[pd.DataFrame] = []
    for df in bloques:
        mapa = {col: canonico[normalizar_texto(col)] for col in df.columns}
        renombrados.append(df.rename(columns=mapa))
    return renombrados


def resolver_columna(
    df: pd.DataFrame,
    alias: Iterable[str],
) -> str | None:
    """Busca en ``df`` la primera columna que coincida con alguno de los alias.

    La comparación es por nombre normalizado, por lo que es insensible a
    mayúsculas, acentos y separadores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame en el que buscar.
    alias : Iterable[str]
        Lista de posibles nombres para el mismo concepto. Se respeta el orden:
        si hay varias coincidencias, se devuelve la del primer alias.

    Returns
    -------
    str | None
        El nombre real de la columna en ``df``, o ``None`` si no se encuentra.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=["RBD", "GPS_LAT", "GPS_LONG"])
    >>> resolver_columna(df, ["latitud", "gps_lat"])
    'GPS_LAT'
    """
    mapa = mapa_normalizado(df)
    for a in alias:
        clave = normalizar_texto(a)
        if clave in mapa:
            return mapa[clave]
    return None


def resolver_columna_req(
    df: pd.DataFrame,
    alias: Iterable[str],
    *,
    concepto: str = "columna",
    contexto: str = "",
) -> str:
    """Igual que :func:`resolver_columna`, pero obligatoria.

    Parameters
    ----------
    df : pandas.DataFrame
    alias : Iterable[str]
    concepto : str, optional
        Nombre legible del concepto buscado (para el mensaje de error).
    contexto : str, optional
        Información extra para el error (por ejemplo, el archivo o el año).

    Returns
    -------
    str
        El nombre real de la columna.

    Raises
    ------
    KeyError
        Si ninguna de las columnas existe en ``df``.
    """
    nombre = resolver_columna(df, alias)
    if nombre is None:
        alias_txt = ", ".join(alias)
        ctx = f" [{contexto}]" if contexto else ""
        raise KeyError(
            f"No se encontró la columna '{concepto}'{ctx}. "
            f"Se buscaron los alias: {alias_txt}. "
            f"Columnas disponibles: {list(df.columns)[:30]}"
        )
    return nombre
