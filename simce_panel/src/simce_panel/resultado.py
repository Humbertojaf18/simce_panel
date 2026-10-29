"""
resultado.py
============

Construcción de las **variables de resultado** de la tesis.

Por qué existe este módulo
--------------------------
La medida usada inicialmente ---promedio POST menos promedio PRE--- mide el
cambio de *nivel* entre dos períodos y mezcla en un solo número el piso
alcanzado en 2022 con el rebote de los años siguientes. Dos establecimientos
con idéntica diferencia de promedios pueden haber seguido trayectorias
opuestas: uno que cayó poco y sigue cayendo, y otro que se desplomó y se está
recuperando rápido.

La recuperación es, por definición, una **trayectoria**. Por eso el fenómeno se
descompone en tres medidas complementarias:

1. :math:`\\text{Caída}_i = \\bar{y}_{i,2022} - \\bar{y}_i^{PRE}`
   — el impacto inicial: cuánto cayó el establecimiento en la primera medición
   posterior al hiato respecto de su nivel pre-pandemia.

2. :math:`\\text{Recuperación}_i = \\hat{\\beta}_i` — la pendiente de una
   regresión lineal del puntaje sobre los años POST. Valores positivos indican
   recuperación sostenida; cercanos a cero, estancamiento.

3. :math:`\\rho_i = (\\bar{y}_{i,\\text{último}} - \\bar{y}_{i,2022}) / |\\text{Caída}_i|`
   — la recuperación relativa, que distingue a quien volvió a su nivel previo
   (:math:`\\rho_i \\geq 1`) de quien solo detuvo el deterioro.

Funciones principales
---------------------
- :func:`promedio_periodo`      : promedio y nº de años válidos por RBD.
- :func:`caida_inicial`         : medida 1.
- :func:`pendiente_recuperacion`: medida 2 (mínimos cuadrados por RBD).
- :func:`recuperacion_relativa` : medida 3.
- :func:`construir_resultados`  : orquesta las tres y arma la tabla final.
- :func:`resumen_resultados`    : estadísticas descriptivas para el informe.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .io_datos import a_numerico

log = logging.getLogger(__name__)

#: Años del período pre-pandemia (últimas cuatro aplicaciones antes del hiato).
ANIOS_PRE: tuple[int, ...] = (2015, 2016, 2017, 2018)

#: Años del período post-pandemia (todas las aplicaciones tras la reanudación).
ANIOS_POST: tuple[int, ...] = (2022, 2023, 2024, 2025)

#: Año de referencia del impacto inicial: primera medición tras el hiato.
ANIO_IMPACTO: int = 2022

#: Mínimo de años válidos exigidos en cada período.
#: En POST se exigen 3 porque la pendiente sobre menos puntos es inestable.
MIN_PRE_DEFECTO: int = 2
MIN_POST_DEFECTO: int = 3


def promedio_periodo(
    df: pd.DataFrame,
    col_puntaje: str,
    anios: tuple[int, ...] | list[int],
    sufijo: str,
    *,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
) -> pd.DataFrame:
    """Promedio del puntaje y número de años válidos por establecimiento.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo (una fila por establecimiento-año), ya saneado.
    col_puntaje : str
        Columna de puntaje (p. ej. ``"prom_mate2m_rbd"``).
    anios : tuple[int, ...] | list[int]
        Años que componen el período.
    sufijo : str
        Sufijo de las columnas creadas (``"pre"`` produce ``puntaje_pre`` y
        ``n_pre``).
    col_rbd, col_anio : str, optional

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD, con ``puntaje_<sufijo>`` y ``n_<sufijo>``.
    """
    d = df[[col_rbd, col_anio, col_puntaje]].copy()
    d["_y"] = a_numerico(d[col_puntaje])
    d = d[d[col_anio].isin(list(anios))].dropna(subset=["_y"])
    out = (d.groupby(col_rbd)
             .agg(**{f"puntaje_{sufijo}": ("_y", "mean"),
                     f"n_{sufijo}": (col_anio, "nunique")}))
    return out


def caida_inicial(
    df: pd.DataFrame,
    col_puntaje: str,
    *,
    anios_pre: tuple[int, ...] = ANIOS_PRE,
    anio_impacto: int = ANIO_IMPACTO,
    min_pre: int = MIN_PRE_DEFECTO,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
) -> pd.DataFrame:
    """Calcula la caída inicial: puntaje del año de impacto menos promedio PRE.

    Es negativa cuando el establecimiento cayó respecto de su nivel previo.

    Parameters
    ----------
    df : pandas.DataFrame
    col_puntaje : str
    anios_pre : tuple[int, ...], optional
    anio_impacto : int, optional
        Primera medición tras el hiato (por defecto 2022).
    min_pre : int, optional
        Mínimo de años válidos en PRE para que el promedio sea aceptable.
    col_rbd, col_anio : str, optional

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD: ``puntaje_pre``, ``n_pre``, ``puntaje_impacto``,
        ``caida_inicial``.
    """
    pre = promedio_periodo(df, col_puntaje, anios_pre, "pre",
                           col_rbd=col_rbd, col_anio=col_anio)

    d = df[[col_rbd, col_anio, col_puntaje]].copy()
    d["_y"] = a_numerico(d[col_puntaje])
    impacto = (d[d[col_anio] == anio_impacto]
               .dropna(subset=["_y"])
               .groupby(col_rbd)["_y"].mean()
               .rename("puntaje_impacto"))

    out = pre.join(impacto, how="outer")
    out["n_pre"] = out["n_pre"].fillna(0).astype(int)

    suficiente = out["n_pre"] >= min_pre
    out["caida_inicial"] = (out["puntaje_impacto"] - out["puntaje_pre"]).where(suficiente)

    n_ok = int(out["caida_inicial"].notna().sum())
    log.info("Caída inicial calculable en %d establecimientos (min_pre=%d).",
             n_ok, min_pre)
    return out


def _pendiente_ols(anios: np.ndarray, valores: np.ndarray) -> float:
    """Pendiente de mínimos cuadrados de ``valores`` sobre ``anios``.

    Se implementa de forma cerrada en lugar de usar ``numpy.polyfit`` porque
    es más rápido sobre miles de grupos pequeños y evita las advertencias de
    condicionamiento que polyfit emite con dos o tres puntos.
    """
    x = anios.astype("float64")
    y = valores.astype("float64")
    x_media = x.mean()
    denom = ((x - x_media) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_media) * (y - y.mean())).sum() / denom)


def pendiente_recuperacion(
    df: pd.DataFrame,
    col_puntaje: str,
    *,
    anios_post: tuple[int, ...] = ANIOS_POST,
    min_post: int = MIN_POST_DEFECTO,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
) -> pd.DataFrame:
    """Estima la recuperación como pendiente anual del puntaje en el período POST.

    La pendiente se interpreta directamente como **puntos SIMCE ganados por
    año** durante el período posterior al hiato.

    Parameters
    ----------
    df : pandas.DataFrame
    col_puntaje : str
    anios_post : tuple[int, ...], optional
    min_post : int, optional
        Mínimo de años válidos exigidos. Con menos de 3 puntos la pendiente
        queda determinada por el ruido de dos mediciones.
    col_rbd, col_anio : str, optional

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD: ``recuperacion`` (pendiente), ``n_post``,
        ``puntaje_post_medio``, ``puntaje_post_ultimo``, ``anio_post_ultimo``.
    """
    d = df[[col_rbd, col_anio, col_puntaje]].copy()
    d["_y"] = a_numerico(d[col_puntaje])
    d = d[d[col_anio].isin(list(anios_post))].dropna(subset=["_y"])
    d = d.sort_values([col_rbd, col_anio])

    filas = []
    for rbd, g in d.groupby(col_rbd, sort=False):
        # Si un año se repitiera, se promedia antes de ajustar la recta.
        g = g.groupby(col_anio, as_index=False)["_y"].mean()
        n = len(g)
        beta = (_pendiente_ols(g[col_anio].to_numpy(), g["_y"].to_numpy())
                if n >= min_post else np.nan)
        filas.append({
            col_rbd: rbd,
            "recuperacion": beta,
            "n_post": n,
            "puntaje_post_medio": float(g["_y"].mean()),
            "puntaje_post_ultimo": float(g["_y"].iloc[-1]),
            "anio_post_ultimo": int(g[col_anio].iloc[-1]),
        })

    out = pd.DataFrame(filas).set_index(col_rbd)
    n_ok = int(out["recuperacion"].notna().sum())
    log.info("Pendiente de recuperación estimable en %d establecimientos "
             "(min_post=%d).", n_ok, min_post)
    return out


def recuperacion_relativa(
    tabla: pd.DataFrame,
    *,
    anio_referencia: int,
    umbral_caida: float = 1.0,
) -> pd.Series:
    """Razón entre lo recuperado y la magnitud de la caída inicial.

    Un valor de 1 indica que el establecimiento recuperó exactamente lo que
    había perdido; por encima de 1, que superó su nivel pre-pandemia.

    El numerador se mide contra un **año de referencia fijo** e igual para
    todos los establecimientos. Usar "el último año disponible de cada uno"
    haría incomparables los valores: un colegio medido por última vez en 2023
    tendría menos tiempo para recuperarse que otro medido en 2025, y la
    diferencia se confundiría con una recuperación más lenta. Los
    establecimientos sin dato en el año de referencia quedan como NaN.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Debe contener ``caida_inicial``, ``puntaje_impacto``,
        ``puntaje_post_ultimo`` y ``anio_post_ultimo``.
    anio_referencia : int
        Año contra el cual se mide la recuperación (normalmente el último del
        período POST).
    umbral_caida : float, optional
        Magnitud mínima de caída (en puntos) para que el cociente tenga
        sentido. Con caídas casi nulas el denominador explota y produce
        valores absurdos, por lo que esos casos quedan como NaN.

    Returns
    -------
    pandas.Series
    """
    tiene_referencia = tabla["anio_post_ultimo"] == anio_referencia
    recuperado = (tabla["puntaje_post_ultimo"] - tabla["puntaje_impacto"]).where(tiene_referencia)

    denom = tabla["caida_inicial"].abs()
    # Corrección del 24-oct-2026: la recuperación relativa solo tiene sentido
    # si hubo caída. Antes se calculaba también para los 865 establecimientos
    # que subieron en 2022 (|caída| >= 1 con signo positivo), lo que mezclaba
    # dos cosas distintas en la mediana.
    valida = tabla["caida_inicial"] <= -umbral_caida
    rho = (recuperado / denom).where(valida)

    n_sin_ref = int((~tiene_referencia & tabla["puntaje_impacto"].notna()).sum())
    if n_sin_ref:
        log.info("%d establecimientos sin medición en %d: recuperación "
                 "relativa no definida (evita comparar horizontes distintos).",
                 n_sin_ref, anio_referencia)
    n_excl = int((~valida & tabla["caida_inicial"].notna()).sum())
    if n_excl:
        log.info("%d establecimientos con caída menor a %.1f punto(s): "
                 "recuperación relativa no definida.", n_excl, umbral_caida)
    return rho.rename("recuperacion_relativa")


def construir_resultados(
    df: pd.DataFrame,
    col_puntaje: str,
    *,
    anios_pre: tuple[int, ...] = ANIOS_PRE,
    anios_post: tuple[int, ...] = ANIOS_POST,
    anio_impacto: int = ANIO_IMPACTO,
    min_pre: int = MIN_PRE_DEFECTO,
    min_post: int = MIN_POST_DEFECTO,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
    incluir_brecha_simple: bool = True,
) -> pd.DataFrame:
    """Construye la tabla completa de variables de resultado por establecimiento.

    Además de las tres medidas principales, adjunta las coordenadas (si el
    panel las trae) y, opcionalmente, la diferencia simple de promedios, que
    se conserva **solo por comparabilidad** con los reportes oficiales de la
    Agencia de Calidad y no como variable de análisis.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo saneado, idealmente ya georreferenciado.
    col_puntaje : str
    anios_pre, anios_post : tuple[int, ...], optional
    anio_impacto : int, optional
    min_pre, min_post : int, optional
    col_rbd, col_anio : str, optional
    incluir_brecha_simple : bool, optional

    Returns
    -------
    pandas.DataFrame
        Una fila por RBD con las tres medidas, sus insumos, los conteos de
        años y las coordenadas.
    """
    caida = caida_inicial(df, col_puntaje, anios_pre=anios_pre,
                          anio_impacto=anio_impacto, min_pre=min_pre,
                          col_rbd=col_rbd, col_anio=col_anio)
    recup = pendiente_recuperacion(df, col_puntaje, anios_post=anios_post,
                                   min_post=min_post, col_rbd=col_rbd,
                                   col_anio=col_anio)

    tabla = caida.join(recup, how="outer")
    tabla["n_post"] = tabla["n_post"].fillna(0).astype(int)
    tabla["recuperacion_relativa"] = recuperacion_relativa(
        tabla, anio_referencia=max(anios_post))

    if incluir_brecha_simple:
        post = promedio_periodo(df, col_puntaje, anios_post, "post_prom",
                                col_rbd=col_rbd, col_anio=col_anio)
        tabla = tabla.join(post[["puntaje_post_prom"]], how="left")
        tabla["brecha_simple"] = tabla["puntaje_post_prom"] - tabla["puntaje_pre"]

    # Coordenadas por RBD (última disponible), si el panel las trae.
    if config.COL_LAT in df.columns and config.COL_LON in df.columns:
        coords = (df.dropna(subset=[config.COL_LAT, config.COL_LON])
                    .sort_values(col_anio)
                    .groupby(col_rbd)[[config.COL_LAT, config.COL_LON]].last())
        tabla = tabla.join(coords, how="left")

    tabla["muestra_analitica"] = (tabla["caida_inicial"].notna()
                                  & tabla["recuperacion"].notna())

    orden = [c for c in ["puntaje_pre", "n_pre", "puntaje_impacto",
                         "caida_inicial", "recuperacion", "n_post",
                         "recuperacion_relativa", "puntaje_post_medio",
                         "puntaje_post_ultimo", "anio_post_ultimo",
                         "puntaje_post_prom", "brecha_simple",
                         config.COL_LAT, config.COL_LON, "muestra_analitica"]
             if c in tabla.columns]
    tabla = tabla[orden].reset_index()

    log.info("Tabla de resultados: %d establecimientos, %d en la muestra "
             "analítica (ambas medidas válidas).",
             len(tabla), int(tabla["muestra_analitica"].sum()))
    return tabla


def resumen_resultados(tabla: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas descriptivas de las variables de resultado.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Salida de :func:`construir_resultados`.

    Returns
    -------
    pandas.DataFrame
        Una fila por medida con n, media, desviación, mínimo, cuartiles y
        máximo, redondeados para el informe.
    """
    medidas = [c for c in ["caida_inicial", "recuperacion",
                           "recuperacion_relativa", "brecha_simple"]
               if c in tabla.columns]
    filas = []
    for m in medidas:
        s = tabla[m].dropna()
        filas.append({
            "medida": m,
            "n": int(s.shape[0]),
            "media": round(float(s.mean()), 2) if len(s) else None,
            "desv": round(float(s.std()), 2) if len(s) else None,
            "min": round(float(s.min()), 2) if len(s) else None,
            "p25": round(float(s.quantile(0.25)), 2) if len(s) else None,
            "mediana": round(float(s.median()), 2) if len(s) else None,
            "p75": round(float(s.quantile(0.75)), 2) if len(s) else None,
            "max": round(float(s.max()), 2) if len(s) else None,
        })
    return pd.DataFrame(filas)


def cobertura_por_umbral(
    df: pd.DataFrame,
    col_puntaje: str,
    *,
    anios_pre: tuple[int, ...] = ANIOS_PRE,
    anios_post: tuple[int, ...] = ANIOS_POST,
    max_umbral: int = 4,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
) -> pd.DataFrame:
    """Cuántos establecimientos sobreviven a cada combinación de umbrales.

    Permite elegir ``min_pre`` y ``min_post`` con evidencia y no por defecto,
    y documentar el compromiso entre precisión de la pendiente y tamaño de la
    muestra.

    Returns
    -------
    pandas.DataFrame
        Filas: ``min_post``; columnas: ``min_pre``; valores: nº de RBD.
    """
    pre = promedio_periodo(df, col_puntaje, anios_pre, "pre",
                           col_rbd=col_rbd, col_anio=col_anio)
    d = df[[col_rbd, col_anio, col_puntaje]].copy()
    d["_y"] = a_numerico(d[col_puntaje])
    post_n = (d[d[col_anio].isin(list(anios_post))].dropna(subset=["_y"])
                .groupby(col_rbd)[col_anio].nunique().rename("n_post"))
    j = pre.join(post_n, how="outer").fillna(0)

    filas = []
    for mpost in range(1, max_umbral + 1):
        fila = {"min_post": mpost}
        for mpre in range(1, max_umbral + 1):
            fila[f"min_pre={mpre}"] = int(((j["n_pre"] >= mpre) &
                                           (j["n_post"] >= mpost)).sum())
        filas.append(fila)
    return pd.DataFrame(filas).set_index("min_post")


# --------------------------------------------------------------------------- #
# Clasificación de rupturas de nivel
# --------------------------------------------------------------------------- #

#: Magnitud (en puntos) a partir de la cual una caída inicial se considera
#: anómala y merece clasificarse. Por debajo, la trayectoria es "normal".
UMBRAL_CAIDA_ANOMALA: float = 40.0

#: Desvío mínimo del período tardío (2023-2025) respecto del PRE para
#: considerar que el cambio de nivel persistió y no fue un año atípico.
UMBRAL_PERSISTENCIA: float = 25.0

#: Años usados para verificar si el cambio persistió.
ANIOS_POST_TARDIO: tuple[int, ...] = (2023, 2024, 2025)


def clasificar_cambio_nivel(
    df: pd.DataFrame,
    tabla: pd.DataFrame,
    col_puntaje: str,
    *,
    anios_pre: tuple[int, ...] = ANIOS_PRE,
    anios_tardio: tuple[int, ...] = ANIOS_POST_TARDIO,
    umbral_caida: float = UMBRAL_CAIDA_ANOMALA,
    umbral_persistencia: float = UMBRAL_PERSISTENCIA,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
) -> pd.DataFrame:
    """Distingue rupturas sostenidas de nivel de valores atípicos de 2022.

    Motivación empírica
    -------------------
    Una parte de los establecimientos presenta caídas o alzas iniciales muy
    grandes (más de 40 puntos). Contra lo que sugiere la intuición, **no se
    explican por el tamaño de la muestra**: en los datos chilenos, la gran
    mayoría de esos casos corresponde a establecimientos con veinte o más
    estudiantes evaluados, y los más extremos superan los doscientos. Tampoco
    se explican por una línea base inestable: la correlación entre la magnitud
    de la caída y la dispersión interna del período PRE es prácticamente nula.

    Lo que los distingue es si el cambio **persiste**. Comparando el nivel de
    2023-2025 con el pre-pandemia se separan dos fenómenos distintos:

    - ``ruptura sostenida``: el establecimiento cambió de nivel de forma
      permanente. No es ruido: es un caso sustantivo, que un promedio de
      recuperación invisibiliza.
    - ``atípico 2022``: la medición de 2022 fue anómala y el establecimiento
      regresó luego a su nivel previo.
    - ``normal``: la caída inicial está dentro de lo esperable.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo saneado.
    tabla : pandas.DataFrame
        Salida de :func:`construir_resultados` (aporta ``caida_inicial``).
    col_puntaje : str
    anios_pre, anios_tardio : tuple[int, ...], optional
    umbral_caida : float, optional
        Magnitud mínima de la caída inicial para clasificar el caso.
    umbral_persistencia : float, optional
        Desvío mínimo del período tardío para considerar que el cambio duró.
    col_rbd, col_anio : str, optional

    Returns
    -------
    pandas.DataFrame
        ``tabla`` con cuatro columnas nuevas: ``puntaje_tardio``,
        ``desvio_tardio``, ``tipo_trayectoria`` y ``ruptura_sostenida``.
    """
    d = df[[col_rbd, col_anio, col_puntaje]].copy()
    d["_y"] = a_numerico(d[col_puntaje])

    pre = (d[d[col_anio].isin(list(anios_pre))].dropna(subset=["_y"])
             .groupby(col_rbd)["_y"].mean())
    tardio = (d[d[col_anio].isin(list(anios_tardio))].dropna(subset=["_y"])
                .groupby(col_rbd)["_y"].mean())

    out = tabla.copy()
    out["puntaje_tardio"] = out[col_rbd].map(tardio)
    out["desvio_tardio"] = out["puntaje_tardio"] - out[col_rbd].map(pre)

    caida = out["caida_inicial"]
    desvio = out["desvio_tardio"]
    mismo_signo = np.sign(caida) == np.sign(desvio)
    persiste = mismo_signo & (desvio.abs() > umbral_persistencia)

    tipo = pd.Series("normal", index=out.index, dtype="object")
    anomala = caida.abs() >= umbral_caida
    tipo[anomala & persiste] = "ruptura sostenida"
    tipo[anomala & ~persiste] = "atípico 2022"
    tipo[caida.isna() | desvio.isna()] = "sin dato"

    out["tipo_trayectoria"] = tipo
    out["ruptura_sostenida"] = tipo == "ruptura sostenida"

    conteo = tipo.value_counts().to_dict()
    log.info("Trayectorias: %s", conteo)
    n_rup = int(out["ruptura_sostenida"].sum())
    if n_rup:
        al_alza = int((out["ruptura_sostenida"] & (caida > 0)).sum())
        log.info("Rupturas sostenidas: %d (%d al alza, %d a la baja).",
                 n_rup, al_alza, n_rup - al_alza)
    return out


def distribucion_rupturas(
    tabla: pd.DataFrame,
    atributos: pd.DataFrame,
    columnas: list[str],
    *,
    col_rbd: str = config.COL_RBD,
) -> dict[str, pd.DataFrame]:
    """Tabula el tipo de trayectoria contra atributos del establecimiento.

    Sirve para responder si las rupturas se concentran en algún segmento
    (dependencia, grupo socioeconómico, ruralidad, región) o se reparten de
    forma pareja, lo que determina si deben tratarse como caso sustantivo o
    como problema de calidad de datos.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Salida de :func:`clasificar_cambio_nivel`.
    atributos : pandas.DataFrame
        Tabla con ``rbd`` y las columnas de atributos (p. ej. el panel ancho).
    columnas : list[str]
        Atributos a tabular.
    col_rbd : str, optional

    Returns
    -------
    dict[str, pandas.DataFrame]
        Por atributo, una tabla de contingencia con la columna adicional
        ``pct_ruptura``.
    """
    j = tabla.merge(atributos, on=col_rbd, how="left")
    salida: dict[str, pd.DataFrame] = {}
    for col in columnas:
        if col not in j.columns:
            log.warning("Atributo '%s' no está en la tabla; se omite.", col)
            continue
        t = pd.crosstab(j[col], j["tipo_trayectoria"])
        rup = t["ruptura sostenida"] if "ruptura sostenida" in t.columns else 0
        t["pct_ruptura"] = (100 * rup / t.sum(axis=1)).round(1)
        salida[col] = t.sort_values("pct_ruptura", ascending=False)
    return salida


# --------------------------------------------------------------------------- #
# Verificación del confundidor composicional
# --------------------------------------------------------------------------- #

#: Orden de las categorías de grupo socioeconómico, de menor a mayor.
#: Permite medir la dirección del cambio y no solo su ocurrencia.
ORDEN_GSE: tuple[str, ...] = ("Bajo", "Medio bajo", "Medio", "Medio alto", "Alto")


def _moda_periodo(serie: pd.Series) -> object:
    """Valor más frecuente de una serie; el primero en caso de empate."""
    s = serie.dropna()
    if s.empty:
        return pd.NA
    modas = s.mode()
    return modas.iloc[0] if len(modas) else pd.NA


def cambio_composicion(
    df: pd.DataFrame,
    *,
    col_gse: str = "cod_grupo_cat",
    col_matricula: str = "nalu_mate2m_rbd",
    anios_pre: tuple[int, ...] = ANIOS_PRE,
    anios_post: tuple[int, ...] = ANIOS_POST,
    col_rbd: str = config.COL_RBD,
    col_anio: str = config.COL_ANIO,
) -> pd.DataFrame:
    """Mide cuánto cambió la composición del establecimiento entre PRE y POST.

    Calcula, por establecimiento, el cambio de grupo socioeconómico (en número
    de categorías y en dirección) y la variación porcentual de la matrícula
    evaluada.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo. Debe traer la columna de GSE ya armonizada a texto
        (``cod_grupo_cat``) y la de matrícula evaluada.
    col_gse, col_matricula : str, optional
    anios_pre, anios_post : tuple[int, ...], optional
    col_rbd, col_anio : str, optional

    Returns
    -------
    pandas.DataFrame
        Por RBD: ``gse_pre``, ``gse_post``, ``delta_gse`` (en categorías,
        negativo si bajó), ``matricula_pre``, ``matricula_post`` y
        ``var_matricula`` (variación porcentual).
    """
    rank = {g: i for i, g in enumerate(ORDEN_GSE)}
    d = df.copy()
    pre_m = d[col_anio].isin(list(anios_pre))
    post_m = d[col_anio].isin(list(anios_post))

    out = pd.DataFrame(index=pd.Index(sorted(d[col_rbd].unique()), name=col_rbd))

    if col_gse in d.columns:
        out["gse_pre"] = d[pre_m].groupby(col_rbd)[col_gse].agg(_moda_periodo)
        out["gse_post"] = d[post_m].groupby(col_rbd)[col_gse].agg(_moda_periodo)
        out["delta_gse"] = (out["gse_post"].map(rank)
                            - out["gse_pre"].map(rank))
    else:
        log.warning("No se encontró la columna de GSE '%s'.", col_gse)

    if col_matricula in d.columns:
        d["_m"] = a_numerico(d[col_matricula])
        out["matricula_pre"] = d[pre_m].groupby(col_rbd)["_m"].mean()
        out["matricula_post"] = d[post_m].groupby(col_rbd)["_m"].mean()
        out["var_matricula"] = (100 * (out["matricula_post"] - out["matricula_pre"])
                                / out["matricula_pre"])
    else:
        log.warning("No se encontró la columna de matrícula '%s'.", col_matricula)

    return out.reset_index()


def verificar_confundidor_composicional(
    tabla: pd.DataFrame,
    composicion: pd.DataFrame,
    *,
    col_rbd: str = config.COL_RBD,
    umbral_matricula: float = -20.0,
) -> dict[str, pd.DataFrame]:
    """Contrasta las rupturas de nivel con los cambios de composición.

    La pregunta que responde: ¿las rupturas sostenidas se explican porque el
    establecimiento cambió de población ---perdió matrícula o bajó de grupo
    socioeconómico--- en lugar de por una pérdida de aprendizaje?

    Si las rupturas coincidieran sistemáticamente con caídas de matrícula o de
    GSE, el fenómeno sería composicional y la interpretación educativa quedaría
    invalidada. Si no, la explicación composicional queda descartada como
    causa principal.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Salida de :func:`clasificar_cambio_nivel`.
    composicion : pandas.DataFrame
        Salida de :func:`cambio_composicion`.
    col_rbd : str, optional
    umbral_matricula : float, optional
        Variación porcentual a partir de la cual se considera pérdida
        relevante de matrícula (por defecto, caída de 20 % o más).

    Returns
    -------
    dict[str, pandas.DataFrame]
        - ``resumen``: por tipo de trayectoria, promedios de cambio de GSE y
          de matrícula, y proporción con pérdida relevante de matrícula.
        - ``tabla_gse``: contingencia entre tipo de trayectoria y dirección
          del cambio de GSE.
        - ``correlaciones``: correlación entre la caída inicial y cada
          variable de composición, en la muestra completa y entre las rupturas.
    """
    j = tabla.merge(composicion, on=col_rbd, how="left")
    salida: dict[str, pd.DataFrame] = {}

    agg: dict[str, tuple] = {"n": ("tipo_trayectoria", "size")}
    if "delta_gse" in j.columns:
        agg["delta_gse_medio"] = ("delta_gse", "mean")
        agg["pct_bajo_gse"] = ("delta_gse", lambda s: 100 * (s < 0).mean())
    if "var_matricula" in j.columns:
        agg["var_matricula_mediana"] = ("var_matricula", "median")
        agg["pct_perdio_matricula"] = (
            "var_matricula", lambda s: 100 * (s <= umbral_matricula).mean())

    resumen = j.groupby("tipo_trayectoria").agg(**agg).round(2)
    salida["resumen"] = resumen

    if "delta_gse" in j.columns:
        direccion = pd.cut(
            j["delta_gse"], bins=[-99, -0.5, 0.5, 99],
            labels=["bajó de GSE", "sin cambio", "subió de GSE"])
        t = pd.crosstab(j["tipo_trayectoria"], direccion, normalize="index")
        salida["tabla_gse"] = (100 * t).round(1)

    filas = []
    rup = j[j.get("ruptura_sostenida", False) == True]  # noqa: E712
    for var in ("delta_gse", "var_matricula"):
        if var not in j.columns:
            continue
        filas.append({
            "variable": var,
            "corr_muestra_completa": round(
                float(j["caida_inicial"].corr(j[var])), 3),
            "corr_solo_rupturas": round(
                float(rup["caida_inicial"].corr(rup[var])), 3) if len(rup) > 2 else None,
        })
    salida["correlaciones"] = pd.DataFrame(filas)

    return salida
