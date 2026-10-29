"""
unificacion.py
==============

Unificación longitudinal de los archivos SIMCE por nivel de agregación.

Para cada nivel (``comuna``, ``deprov``, ``region``, ``rbd``) recorre todas
las carpetas de año disponibles, localiza el CSV correspondiente, le añade una
columna ``año`` y apila (concatena) todos los años en un único DataFrame.

El emparejamiento de archivos es *dinámico*: no se asume un nombre fijo, sino
que se busca dentro de cada carpeta de año el CSV cuyo nombre contenga el
token del nivel. Esto tolera variaciones reales como
``simce2m2016_comuna_publica_fina.csv`` (sin la "l"),
``simce2m2023_comuna_público_final.csv`` o
``simce2m2025_comuna_preliminar.csv``.

Funciones principales
---------------------
- :func:`buscar_archivo_nivel` : encuentra el CSV de un nivel en una carpeta.
- :func:`mapear_archivos`       : {nivel: [(anio, ruta), ...]} para todos los años.
- :func:`unificar_nivel`        : apila todos los años de un nivel.
- :func:`unificar_todos`        : ejecuta la unificación de los 4 niveles.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config
from .columnas import normalizar_texto, resolver_columna, unificar_nombres_columnas
from .io_datos import extraer_anio, leer_csv, listar_subcarpetas_con_anio

log = logging.getLogger(__name__)

# Tokens que identifican cada nivel dentro del nombre de archivo.
# Se comparan con el nombre normalizado del archivo.
_TOKENS_NIVEL: dict[str, str] = {
    "comuna": "comuna",
    "deprov": "deprov",
    "region": "region",
    "rbd": "rbd",
}


def buscar_archivo_nivel(carpeta_anio: str | Path, nivel: str) -> Path | None:
    """Busca dentro de una carpeta de año el CSV correspondiente a un nivel.

    Reglas de selección:

    1. Solo archivos ``.csv`` (ignora las glosas ``.xlsx``).
    2. El nombre normalizado debe contener el token del nivel.
    3. Para evitar falsos positivos con ``rbd`` (token corto), se exige que el
       token aparezca delimitado (precedido y seguido por un carácter no
       alfanumérico o por el inicio/fin del nombre normalizado-con-separadores).

    Parameters
    ----------
    carpeta_anio : str | Path
        Carpeta de un año específico (p. ej. ``Simce 2° Medio 2015``).
    nivel : str
        Uno de ``comuna``, ``deprov``, ``region``, ``rbd``.

    Returns
    -------
    Path | None
        Ruta del CSV encontrado, o ``None`` si no hay coincidencia.
    """
    carpeta_anio = Path(carpeta_anio)
    token = _TOKENS_NIVEL[nivel]
    candidatos: list[Path] = []
    for archivo in sorted(carpeta_anio.glob("*.csv")):
        # Normaliza conservando un separador para poder delimitar el token.
        base = normalizar_texto(archivo.stem)  # p. ej. 'simce2m2015comunapublicafinal'
        if re.search(rf"(^|[^a-z]){token}([^a-z]|$)", base):
            candidatos.append(archivo)
        elif token in base:
            # Coincidencia laxa (sin delimitador) como respaldo.
            candidatos.append(archivo)

    if not candidatos:
        return None
    if len(candidatos) > 1:
        log.warning(
            "Varios archivos coinciden con el nivel '%s' en %s: %s. "
            "Se usa el primero.",
            nivel, carpeta_anio.name, [c.name for c in candidatos],
        )
    return candidatos[0]


def mapear_archivos(
    cfg: config.Config,
    niveles: tuple[str, ...] = config.NIVELES,
) -> dict[str, list[tuple[int, Path]]]:
    """Construye el mapa de archivos disponibles por nivel y año.

    Parameters
    ----------
    cfg : config.Config
    niveles : tuple[str, ...], optional

    Returns
    -------
    dict[str, list[tuple[int, Path]]]
        Para cada nivel, lista ordenada de ``(anio, ruta_csv)``.
    """
    carpetas = listar_subcarpetas_con_anio(cfg.simce_dir)
    mapa: dict[str, list[tuple[int, Path]]] = {n: [] for n in niveles}
    for anio, carpeta in carpetas:
        for nivel in niveles:
            ruta = buscar_archivo_nivel(carpeta, nivel)
            if ruta is not None:
                mapa[nivel].append((anio, ruta))
            else:
                log.warning("No se encontró archivo de '%s' para el año %s.",
                            nivel, anio)
    return mapa


def _resolver_anio_fila(df: pd.DataFrame, anio_archivo: int) -> pd.Series:
    """Determina el año de cada fila.

    Prioriza la columna interna de año del archivo (``agno``/``año``) y, si no
    existe o tiene valores faltantes, completa con el año extraído del nombre
    del archivo. Así la columna ``año`` queda siempre poblada.
    """
    col_anio = resolver_columna(df, config.ALIAS_COLUMNAS["anio"])
    if col_anio is not None:
        serie = pd.to_numeric(df[col_anio], errors="coerce")
        # Completa NaN con el año del nombre del archivo.
        serie = serie.fillna(anio_archivo).astype("int64")
        return serie
    return pd.Series([anio_archivo] * len(df), index=df.index, dtype="int64")


def unificar_nivel(
    cfg: config.Config,
    nivel: str,
    archivos: list[tuple[int, Path]] | None = None,
) -> pd.DataFrame:
    """Apila todos los años disponibles de un nivel en un único DataFrame.

    Cada bloque anual recibe una columna ``año`` (constante de :data:`config.COL_ANIO`)
    ubicada como primera columna. La concatenación alinea por unión de columnas,
    de modo que variables presentes solo en algunos años quedan como NaN en los
    demás (comportamiento deseado para un panel longitudinal).

    Parameters
    ----------
    cfg : config.Config
    nivel : str
        Uno de ``comuna``, ``deprov``, ``region``, ``rbd``.
    archivos : list[tuple[int, Path]] | None, optional
        Lista ``(anio, ruta)``. Si es ``None``, se calcula con
        :func:`mapear_archivos`.

    Returns
    -------
    pandas.DataFrame
        Panel unificado del nivel con la columna ``año`` al frente.
    """
    if archivos is None:
        archivos = mapear_archivos(cfg, (nivel,))[nivel]

    bloques: list[pd.DataFrame] = []
    log.info("Unificando nivel '%s' (%d años)...", nivel, len(archivos))
    for anio, ruta in archivos:
        df = leer_csv(ruta, sep=cfg.sep, encoding=cfg.encoding)
        df.insert(0, config.COL_ANIO, _resolver_anio_fila(df, anio))
        log.info("  %s -> %d filas, %d columnas", anio, len(df), df.shape[1])
        bloques.append(df)

    if not bloques:
        log.warning("No hay archivos para el nivel '%s'.", nivel)
        return pd.DataFrame()

    # Homologa nombres equivalentes (p. ej. 'RBD' vs 'rbd') para que cada
    # concepto sea UNA sola columna tras la concatenación.
    bloques = unificar_nombres_columnas(bloques)
    panel = pd.concat(bloques, ignore_index=True, sort=False)
    # Reordena para dejar 'año' como primera columna (insert ya lo hace por
    # bloque, pero concat puede reordenar al unir columnas distintas).
    cols = [config.COL_ANIO] + [c for c in panel.columns if c != config.COL_ANIO]
    panel = panel[cols]
    log.info("  => Panel '%s': %d filas, %d columnas",
             nivel, len(panel), panel.shape[1])
    return panel


def unificar_todos(
    cfg: config.Config,
    niveles: tuple[str, ...] = config.NIVELES,
    guardar: bool = True,
) -> dict[str, pd.DataFrame]:
    """Unifica los cuatro niveles y, opcionalmente, los guarda en disco.

    Parameters
    ----------
    cfg : config.Config
    niveles : tuple[str, ...], optional
    guardar : bool, optional
        Si ``True`` (por defecto) escribe cada panel en ``Unificados`` usando
        los nombres de :data:`config.NOMBRES_SALIDA`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``{nivel: panel_unificado}``.
    """
    from .io_datos import escribir_csv  # import local para evitar ciclos

    mapa = mapear_archivos(cfg, niveles)
    paneles: dict[str, pd.DataFrame] = {}
    for nivel in niveles:
        panel = unificar_nivel(cfg, nivel, mapa[nivel])
        paneles[nivel] = panel
        if guardar and not panel.empty:
            ruta = cfg.ruta_salida(config.NOMBRES_SALIDA[nivel])
            escribir_csv(panel, ruta, sep=cfg.sep, encoding=cfg.encoding)
            log.info("  Guardado: %s", ruta)
    return paneles
