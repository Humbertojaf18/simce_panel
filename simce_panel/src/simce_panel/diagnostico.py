"""
diagnostico.py
==============

Revisiones de calidad del panel RBD antes del análisis espacial y el
modelamiento. Responde a tres preguntas metodológicas:

1. **¿Hay puntajes suprimidos o valores anómalos?** En las bases públicas
   SIMCE, los establecimientos con muy pocos estudiantes tienen el puntaje
   *oculto*; las columnas ``marca_*`` indican por qué. Si la supresión se
   codificó como un número (0, 99, etc.) en lugar de quedar vacía,
   contaminaría los promedios.
2. **¿Cuántos años respaldan cada brecha?** El target POST−PRE es más confiable
   si se calcula con varios años en cada periodo. Aquí se cuentan los años por
   establecimiento y se permite exigir un mínimo.
3. **¿Cambió el número de regiones?** Ñuble se separó de Biobío en 2018, justo
   sobre el corte pre/post.

Funciones principales
---------------------
- :func:`detectar_columnas_puntaje`  : localiza las columnas ``prom_*2m_*``.
- :func:`inspeccionar_puntajes`      : rango, NaN y valores fuera de rango.
- :func:`cruce_marca_puntaje`        : relación entre ``marca_*`` y el puntaje.
- :func:`cobertura_pre_post`         : años por RBD en cada periodo.
- :func:`resumen_umbral_minimo`      : establecimientos según mínimo de años.
- :func:`construir_target`           : target POST−PRE con n_pre/n_post y filtro.
- :func:`conteo_regiones_por_anio`   : regiones distintas por año (Ñuble).
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config
from .columnas import normalizar_texto, resolver_columna
from .io_datos import a_numerico

log = logging.getLogger(__name__)

#: Rango plausible de un puntaje SIMCE 2° medio, usado como **filtro de
#: diagnóstico** para detectar códigos de supresión colados como puntaje.
#:
#: Sobre el límite superior. La escala SIMCE es relativa: se estandariza con
#: promedio 250 y desviación estándar 50, y NO define valores máximos ni
#: mínimos establecidos. El valor 400 que aparece en la documentación de los
#: Estándares de Aprendizaje es el tope del eje con que la Agencia reporta los
#: niveles de logro, no un techo de la escala. Un promedio de establecimiento
#: de 420 equivale a 3,4 desviaciones estándar sobre la media: excepcional,
#: pero perfectamente posible en colegios de alto desempeño.
#:
#: Por eso el límite superior se fija en 450 y no en 400: cortar en 400
#: marcaría como anómalos a establecimientos reales. El trabajo útil de este
#: rango está en el límite INFERIOR, donde viven los ceros de supresión.
RANGO_PUNTAJE_VALIDO: tuple[float, float] = (100.0, 450.0)

#: Fragmentos (en minúsculas, sin acentos) que en la columna ``marca_*``
#: identifican resultados que la Agencia considera NO representativos y que,
#: por tanto, conviene excluir del target aunque traigan un número.
PATRONES_NO_REPRESENTATIVO: tuple[str, ...] = (
    "no representativ",   # "Resultados no representativos"
    "ajenas",             # "Ajenas a la Agencia / al Establecimiento"
    "insuficient",        # "Alumnos insuficientes"
)


def columnas_por_fragmentos(df: pd.DataFrame, *fragmentos: str) -> list[str]:
    """Devuelve las columnas cuyo nombre normalizado contiene TODOS los fragmentos.

    Parameters
    ----------
    df : pandas.DataFrame
    *fragmentos : str
        Trozos a buscar (ya se normalizan internamente).

    Returns
    -------
    list[str]

    Examples
    --------
    >>> columnas_por_fragmentos(df, "prom", "mate")   # -> ['prom_mate2m_rbd']
    """
    frags = [normalizar_texto(f) for f in fragmentos]
    return [c for c in df.columns
            if all(f in normalizar_texto(c) for f in frags)]


def detectar_columnas_puntaje(df: pd.DataFrame) -> list[str]:
    """Localiza las columnas de promedio de puntaje (``prom_*2m_*``).

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list[str]
        Nombres de las columnas de puntaje promedio detectadas.
    """
    cols = columnas_por_fragmentos(df, "prom", "2m")
    # Excluye columnas que claramente no son puntaje (por si acaso).
    return [c for c in cols if "fecha" not in normalizar_texto(c)]


def inspeccionar_puntajes(
    df: pd.DataFrame,
    cols_puntaje: list[str] | None = None,
    rango_valido: tuple[float, float] = RANGO_PUNTAJE_VALIDO,
) -> pd.DataFrame:
    """Inspecciona cada columna de puntaje buscando NaN y valores fuera de rango.

    Convierte cada columna a numérico (tolerando coma decimal) y reporta cuántos
    valores son válidos, cuántos quedaron como NaN y cuántos caen fuera del
    rango plausible (posibles códigos de supresión o sentinelas).

    Parameters
    ----------
    df : pandas.DataFrame
    cols_puntaje : list[str] | None, optional
        Columnas a inspeccionar. Si es ``None``, se autodetectan.
    rango_valido : tuple[float, float], optional
        Rango plausible ``(mínimo, máximo)`` de un puntaje válido.

    Returns
    -------
    pandas.DataFrame
        Una fila por columna con: ``n_total``, ``n_numerico``, ``n_nan``,
        ``min``, ``max``, ``n_fuera_rango`` y ``ejemplos_fuera_rango``.
    """
    if cols_puntaje is None:
        cols_puntaje = detectar_columnas_puntaje(df)

    lo, hi = rango_valido
    filas = []
    for col in cols_puntaje:
        s = a_numerico(df[col])
        validos = s.dropna()
        fuera = validos[(validos < lo) | (validos > hi)]
        filas.append({
            "columna": col,
            "n_total": len(s),
            "n_numerico": int(validos.shape[0]),
            "n_nan": int(s.isna().sum()),
            "min": round(float(validos.min()), 2) if len(validos) else None,
            "max": round(float(validos.max()), 2) if len(validos) else None,
            "n_fuera_rango": int(fuera.shape[0]),
            "ejemplos_fuera_rango": sorted(fuera.unique().tolist())[:5],
        })
    return pd.DataFrame(filas)


def cruce_marca_puntaje(
    df: pd.DataFrame,
    materia: str = "mate",
) -> pd.DataFrame:
    """Cruza la columna ``marca_<materia>`` con la presencia del puntaje.

    Para cada valor de la marca, indica cuántas filas tienen puntaje presente o
    ausente y el rango del puntaje. Permite verificar que las filas marcadas
    como suprimidas efectivamente no traen un número válido (y, si lo traen,
    detectarlo).

    Parameters
    ----------
    df : pandas.DataFrame
    materia : str, optional
        ``"lect"``, ``"mate"``, ``"soc"`` o ``"nat"``.

    Returns
    -------
    pandas.DataFrame
        Por cada valor de marca: ``n_filas``, ``con_puntaje``, ``sin_puntaje``,
        ``min_puntaje``, ``max_puntaje``. Vacío si no existen las columnas.
    """
    col_prom = (columnas_por_fragmentos(df, "prom", materia, "2m") or [None])[0]
    col_marca = (columnas_por_fragmentos(df, "marca", materia, "2m") or [None])[0]
    # Evita confundir 'marca' con 'marcadif' o 'marcapalu'.
    marcas_simples = [c for c in columnas_por_fragmentos(df, "marca", materia, "2m")
                      if "dif" not in normalizar_texto(c)
                      and "palu" not in normalizar_texto(c)]
    if marcas_simples:
        col_marca = marcas_simples[0]

    if col_prom is None or col_marca is None:
        log.warning("No se hallaron columnas prom/marca para '%s' "
                    "(prom=%s, marca=%s).", materia, col_prom, col_marca)
        return pd.DataFrame()

    tmp = pd.DataFrame({
        "marca": df[col_marca].astype("string"),
        "puntaje": a_numerico(df[col_prom]),
    })
    out = (tmp.groupby("marca", dropna=False)
           .agg(n_filas=("puntaje", "size"),
                con_puntaje=("puntaje", lambda s: int(s.notna().sum())),
                sin_puntaje=("puntaje", lambda s: int(s.isna().sum())),
                min_puntaje=("puntaje", "min"),
                max_puntaje=("puntaje", "max"))
           .reset_index()
           .sort_values("n_filas", ascending=False))
    out.attrs["col_prom"] = col_prom
    out.attrs["col_marca"] = col_marca
    return out


def cobertura_pre_post(
    df: pd.DataFrame,
    col_puntaje: str,
    pre: list[int],
    post: list[int],
    col_rbd: str = config.COL_RBD,
) -> pd.DataFrame:
    """Cuenta cuántos años con puntaje válido tiene cada RBD en cada periodo.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel RBD.
    col_puntaje : str
        Columna de puntaje a usar (p. ej. ``"prom_mate2m_rbd"``).
    pre, post : list[int]
        Años de cada periodo.
    col_rbd : str, optional

    Returns
    -------
    pandas.DataFrame
        Por RBD: ``n_pre``, ``n_post`` (nº de años con puntaje válido).
    """
    d = df.copy()
    d["_p"] = a_numerico(d[col_puntaje])
    d = d.dropna(subset=["_p"])
    n_pre = (d[d[config.COL_ANIO].isin(pre)]
             .groupby(col_rbd)[config.COL_ANIO].nunique().rename("n_pre"))
    n_post = (d[d[config.COL_ANIO].isin(post)]
              .groupby(col_rbd)[config.COL_ANIO].nunique().rename("n_post"))
    cob = pd.concat([n_pre, n_post], axis=1).fillna(0).astype(int).reset_index()
    return cob


def resumen_umbral_minimo(
    cobertura: pd.DataFrame,
    max_umbral: int = 4,
) -> pd.DataFrame:
    """Cuenta cuántos RBD sobreviven según el mínimo de años exigido.

    Parameters
    ----------
    cobertura : pandas.DataFrame
        Salida de :func:`cobertura_pre_post`.
    max_umbral : int, optional
        Umbral máximo a evaluar.

    Returns
    -------
    pandas.DataFrame
        Por cada ``min_anios`` (1..max): ``rbd_validos`` (con n_pre>=min y
        n_post>=min).
    """
    filas = []
    for t in range(1, max_umbral + 1):
        n = int(((cobertura["n_pre"] >= t) & (cobertura["n_post"] >= t)).sum())
        filas.append({"min_anios_por_periodo": t, "rbd_validos": n})
    return pd.DataFrame(filas)


def construir_target(
    df: pd.DataFrame,
    col_puntaje: str,
    pre: list[int],
    post: list[int],
    *,
    min_pre: int = 1,
    min_post: int = 1,
    col_rbd: str = config.COL_RBD,
) -> pd.DataFrame:
    """Construye el target POST−PRE por RBD con conteo de años y filtro mínimo.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel RBD (idealmente ya enriquecido con coordenadas).
    col_puntaje : str
        Columna de puntaje (p. ej. ``"prom_mate2m_rbd"``).
    pre, post : list[int]
        Años de cada periodo.
    min_pre, min_post : int, optional
        Mínimo de años con puntaje válido requeridos en cada periodo para que
        la brecha se considere válida (por defecto 1).
    col_rbd : str, optional

    Returns
    -------
    pandas.DataFrame
        Por RBD: ``puntaje_pre``, ``n_pre``, ``puntaje_post``, ``n_post``,
        ``brecha_recuperacion``, ``cumple_minimo`` y, si existen,
        ``latitud``/``longitud``. La brecha es NaN si no se cumple el mínimo.
    """
    d = df.copy()
    d["_p"] = a_numerico(d[col_puntaje])

    def _agg(anios: list[int], sufijo: str) -> pd.DataFrame:
        sub = d[d[config.COL_ANIO].isin(anios)].dropna(subset=["_p"])
        return (sub.groupby(col_rbd)["_p"]
                .agg(["mean", "nunique"])
                .rename(columns={"mean": f"puntaje_{sufijo}",
                                 "nunique": f"n_{sufijo}"}))

    pre_t = _agg(pre, "pre")
    post_t = _agg(post, "post")
    target = pd.concat([pre_t, post_t], axis=1)
    for c in ("n_pre", "n_post"):
        if c in target.columns:
            target[c] = target[c].fillna(0).astype(int)

    target["cumple_minimo"] = (
        (target.get("n_pre", 0) >= min_pre) & (target.get("n_post", 0) >= min_post)
    )
    target["brecha_recuperacion"] = (
        target["puntaje_post"] - target["puntaje_pre"]
    ).where(target["cumple_minimo"])

    # Coordenadas por RBD (última disponible) si el panel las tiene.
    if config.COL_LAT in d.columns and config.COL_LON in d.columns:
        coords = (d.dropna(subset=[config.COL_LAT, config.COL_LON])
                  .sort_values(config.COL_ANIO)
                  .groupby(col_rbd)[[config.COL_LAT, config.COL_LON]].last())
        target = target.join(coords)

    return target.reset_index()


def sanear_puntajes(
    df: pd.DataFrame,
    materias: tuple[str, ...] = ("lect", "mate", "soc", "nat"),
    *,
    min_valido: float = 100.0,
    max_valido: float | None = None,
    patrones_no_repr: tuple[str, ...] = PATRONES_NO_REPRESENTATIVO,
    inplace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sanea las columnas de puntaje dejando en NaN los valores no confiables.

    Aplica, por cada materia, tres limpiezas sobre la columna ``prom_<materia>``:

    1. **Sentinelas bajos**: puntajes ``< min_valido`` (p. ej. los ``0.0`` que
       son códigos de supresión colados como número) → NaN.
    2. **Valores altos** (opcional): si se pasa ``max_valido``, los puntajes
       ``> max_valido`` → NaN. Por defecto es ``None`` y los valores altos se
       **conservan** para inspección manual (pueden ser colegios reales).
    3. **No representativos**: filas cuya ``marca_<materia>`` contenga alguno de
       los ``patrones_no_repr`` (resultados que la Agencia marca como no
       comparables) → NaN, aunque traigan un número.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel RBD.
    materias : tuple[str, ...], optional
        Materias a sanear (``"lect"``, ``"mate"``, ``"soc"``, ``"nat"``). Las que
        no existan en el panel se omiten.
    min_valido : float, optional
        Umbral inferior de puntaje válido (por defecto 100).
    max_valido : float | None, optional
        Umbral superior. ``None`` (por defecto) conserva los valores altos.
    patrones_no_repr : tuple[str, ...], optional
        Fragmentos de texto que marcan resultados no representativos.
    inplace : bool, optional
        Si ``True`` modifica ``df`` directamente; si ``False`` (por defecto)
        trabaja sobre una copia.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(df_saneado, reporte)``. El reporte tiene una fila por materia con
        cuántos valores se pusieron en NaN por cada criterio.
    """
    out = df if inplace else df.copy()
    filas_reporte = []

    for materia in materias:
        col = (columnas_por_fragmentos(out, "prom", materia, "2m") or [None])[0]
        if col is None:
            continue
        marcas = [c for c in columnas_por_fragmentos(out, "marca", materia, "2m")
                  if "dif" not in normalizar_texto(c)
                  and "palu" not in normalizar_texto(c)]
        # Se prefiere la marca exacta del puntaje: en un panel ya estandarizado
        # conviven 'marca_<m>2m_rbd' y sus capas '_desc' / '_cat'.
        exacta = f"marca_{materia}2m_rbd"
        col_marca = exacta if exacta in out.columns else (marcas[0] if marcas else None)

        s = a_numerico(out[col])
        n_inicial = int(s.notna().sum())

        n_bajos = int((s < min_valido).sum())
        s = s.where(s >= min_valido)

        n_altos = 0
        if max_valido is not None:
            n_altos = int((s > max_valido).sum())
            s = s.where(s <= max_valido)

        n_no_repr = 0
        if col_marca is not None:
            m = out[col_marca].astype("string").str.lower()
            no_repr = m.str.contains("|".join(patrones_no_repr), na=False)
            n_no_repr = int((no_repr & s.notna()).sum())
            s = s.where(~no_repr)

        out[col] = s
        filas_reporte.append({
            "materia": materia,
            "columna": col,
            "n_validos_inicial": n_inicial,
            "n_bajos_a_nan": n_bajos,
            "n_altos_a_nan": n_altos,
            "n_no_repr_a_nan": n_no_repr,
            "n_validos_final": int(s.notna().sum()),
        })

    reporte = pd.DataFrame(filas_reporte)
    return out, reporte


def conteo_regiones_por_anio(df_region: pd.DataFrame) -> pd.DataFrame:
    """Cuenta regiones distintas por año y detecta altas/bajas (caso Ñuble).

    Parameters
    ----------
    df_region : pandas.DataFrame
        Panel a nivel región.

    Returns
    -------
    pandas.DataFrame
        Por año: ``n_regiones`` y ``codigos`` (lista ordenada de códigos).
    """
    col_reg = resolver_columna(df_region, config.ALIAS_COLUMNAS["region"]) \
        or (columnas_por_fragmentos(df_region, "cod", "reg") or [None])[0]
    if col_reg is None:
        log.warning("No se encontró columna de código de región.")
        return pd.DataFrame()

    filas = []
    for anio, g in df_region.groupby(config.COL_ANIO):
        codigos = sorted(pd.to_numeric(g[col_reg], errors="coerce")
                         .dropna().astype(int).unique().tolist())
        filas.append({"año": anio, "n_regiones": len(codigos), "codigos": codigos})
    return pd.DataFrame(filas).sort_values("año").reset_index(drop=True)
