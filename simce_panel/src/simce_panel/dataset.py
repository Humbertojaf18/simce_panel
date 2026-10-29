"""
dataset.py
==========

Ensamblaje del **dataset analítico**: la tabla de una fila por establecimiento
que alimenta el análisis espacial y el modelamiento.

Combina tres fuentes ya construidas:

1. Las variables de resultado (:mod:`resultado`): caída inicial, recuperación
   y recuperación relativa, más la clasificación de trayectoria.
2. Las covariables del panel ancho: grupo socioeconómico, dependencia,
   ruralidad, región, comuna y matrícula.
3. Variables **geográficas derivadas**, calculadas aquí a partir de las
   coordenadas.

Por qué variables geográficas derivadas y no las coordenadas crudas
-------------------------------------------------------------------
Un modelo de árboles puede particionar latitud y longitud hasta memorizar la
ubicación de cada establecimiento. En ese caso la contribución SHAP de las
coordenadas no admite lectura como "efecto del entorno geográfico": solo
indica que el modelo aprendió dónde está cada colegio. Las variables de este
módulo son interpretables por sí mismas ---densidad de la red escolar,
aislamiento, posición dentro de la comuna, distancia al centro nacional--- y
permiten enunciar hipótesis comprobables sobre el territorio.

Funciones principales
---------------------
- :func:`variables_geograficas` : densidad, aislamiento y distancias.
- :func:`matricula_periodo`     : matrícula promedio en PRE y POST.
- :func:`ensamblar_dataset`     : arma la tabla analítica completa.
- :func:`resumen_por_grupo`     : descriptivos de las medidas por covariable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Radio de Tierra en kilómetros, para convertir distancias angulares.
RADIO_TIERRA_KM: float = 6371.0

#: Coordenadas de referencia nacional (Plaza de Armas de Santiago). Se usan
#: para una variable de centralidad respecto del núcleo urbano principal del
#: país, no como juicio sobre la importancia de otras ciudades.
REFERENCIA_NACIONAL: tuple[float, float] = (-33.4372, -70.6506)

#: Radios (km) en los que se cuenta la densidad de establecimientos.
RADIOS_DENSIDAD_KM: tuple[float, ...] = (5.0, 10.0)

#: Vecinos considerados para la medida de aislamiento.
K_VECINOS: int = 5


def _a_radianes(df: pd.DataFrame, col_lat: str, col_lon: str) -> np.ndarray:
    """Matriz (n, 2) de coordenadas en radianes, para métricas haversine."""
    return np.radians(df[[col_lat, col_lon]].to_numpy(dtype="float64"))


def variables_geograficas(
    df: pd.DataFrame,
    *,
    col_rbd: str = "rbd",
    col_lat: str = "latitud",
    col_lon: str = "longitud",
    col_comuna: str = "comuna_dir",
    radios_km: tuple[float, ...] = RADIOS_DENSIDAD_KM,
    k_vecinos: int = K_VECINOS,
    referencia: tuple[float, float] = REFERENCIA_NACIONAL,
) -> pd.DataFrame:
    """Calcula variables geográficas interpretables a partir de las coordenadas.

    Todas las distancias se calculan con la métrica haversine sobre la esfera,
    no con distancia euclidiana sobre grados: en Chile, un grado de longitud
    mide unos 104 km en Arica y 62 km en Punta Arenas, de modo que tratar los
    grados como un plano distorsionaría sistemáticamente las medidas según la
    latitud, justo en la dimensión que el estudio quiere analizar.

    Variables producidas
    --------------------
    - ``densidad_<r>km`` : número de establecimientos dentro de un radio de
      ``r`` km (sin contarse a sí mismo). Aproxima la densidad de la oferta
      escolar y, por tanto, la intensidad de la competencia entre colegios.
    - ``dist_vecino_km`` : distancia al establecimiento más cercano.
    - ``dist_k_vecinos_km`` : distancia media a los ``k`` más cercanos, una
      medida de aislamiento más estable que la anterior.
    - ``dist_centroide_comuna_km`` : distancia al centroide de los
      establecimientos de su comuna, que distingue a los colegios céntricos de
      los periféricos dentro de una misma unidad administrativa.
    - ``dist_referencia_km`` : distancia al núcleo urbano nacional.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con RBD y coordenadas.
    col_rbd, col_lat, col_lon, col_comuna : str, optional
    radios_km : tuple[float, ...], optional
    k_vecinos : int, optional
    referencia : tuple[float, float], optional

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD, con las variables geográficas. Los establecimientos
        sin coordenadas quedan con NaN.
    """
    from sklearn.neighbors import BallTree

    geo = df[[col_rbd, col_lat, col_lon]].dropna(subset=[col_lat, col_lon]).copy()
    if geo.empty:
        log.warning("No hay establecimientos con coordenadas.")
        return pd.DataFrame(index=pd.Index([], name=col_rbd))

    coords = _a_radianes(geo, col_lat, col_lon)
    arbol = BallTree(coords, metric="haversine")

    out = pd.DataFrame(index=pd.Index(geo[col_rbd].to_numpy(), name=col_rbd))

    # Densidad: vecinos dentro de cada radio, descontando el propio punto.
    for r in radios_km:
        conteo = arbol.query_radius(coords, r=r / RADIO_TIERRA_KM, count_only=True)
        out[f"densidad_{int(r)}km"] = conteo - 1

    # Aislamiento: distancia al vecino más cercano y a los k más cercanos.
    k = min(k_vecinos + 1, len(geo))
    dist, _ = arbol.query(coords, k=k)
    dist_km = dist * RADIO_TIERRA_KM
    out["dist_vecino_km"] = dist_km[:, 1] if k > 1 else np.nan
    out["dist_k_vecinos_km"] = dist_km[:, 1:].mean(axis=1) if k > 1 else np.nan

    # Posición dentro de la comuna.
    if col_comuna in df.columns:
        g = df[[col_rbd, col_lat, col_lon, col_comuna]].dropna(
            subset=[col_lat, col_lon, col_comuna])
        centro = g.groupby(col_comuna)[[col_lat, col_lon]].mean()
        g = g.join(centro, on=col_comuna, rsuffix="_c")
        out["dist_centroide_comuna_km"] = pd.Series(
            _haversine_km(g[col_lat].to_numpy(), g[col_lon].to_numpy(),
                          g[f"{col_lat}_c"].to_numpy(), g[f"{col_lon}_c"].to_numpy()),
            index=g[col_rbd].to_numpy())
    else:
        log.info("Sin columna de comuna; se omite la distancia al centroide.")

    # Centralidad nacional.
    out["dist_referencia_km"] = _haversine_km(
        geo[col_lat].to_numpy(), geo[col_lon].to_numpy(),
        np.full(len(geo), referencia[0]), np.full(len(geo), referencia[1]))

    log.info("Variables geográficas calculadas para %d establecimientos.", len(out))
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia haversine en kilómetros entre pares de coordenadas."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def matricula_periodo(
    ancho: pd.DataFrame,
    *,
    materia: str = "mate",
    anios_pre: tuple[int, ...] = (2015, 2016, 2017, 2018),
    anios_post: tuple[int, ...] = (2022, 2023, 2024, 2025),
    col_rbd: str = "rbd",
) -> pd.DataFrame:
    """Matrícula evaluada promedio en cada período y su variación.

    Parameters
    ----------
    ancho : pandas.DataFrame
        Panel ancho, con columnas ``nalu_<materia>2m_rbd_<año>``.
    materia : str, optional
    anios_pre, anios_post : tuple[int, ...], optional
    col_rbd : str, optional

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD: ``matricula_pre``, ``matricula_post``,
        ``var_matricula_pct``.
    """
    def _cols(anios):
        return [c for c in (f"nalu_{materia}2m_rbd_{a}" for a in anios)
                if c in ancho.columns]

    cpre, cpost = _cols(anios_pre), _cols(anios_post)
    if not cpre or not cpost:
        log.warning("No se hallaron columnas de matrícula para '%s'.", materia)
        return pd.DataFrame(index=pd.Index([], name=col_rbd))

    out = pd.DataFrame(index=pd.Index(ancho[col_rbd].to_numpy(), name=col_rbd))
    out["matricula_pre"] = ancho[cpre].mean(axis=1).to_numpy()
    out["matricula_post"] = ancho[cpost].mean(axis=1).to_numpy()
    out["var_matricula_pct"] = (
        100 * (out["matricula_post"] - out["matricula_pre"]) / out["matricula_pre"])
    return out


#: Covariables categóricas y continuas que se arrastran del panel ancho.
COVARIABLES_ANCHO: tuple[str, ...] = (
    "nom_rbd", "region_dir", "comuna_dir",
    "cod_depe1_cat", "cod_depe2_cat", "cod_grupo_cat", "cod_rural_rbd_cat",
    "cod_depe1_cambio", "cod_grupo_cambio", "n_anios_presente",
)


def ensamblar_dataset(
    resultados: pd.DataFrame,
    ancho: pd.DataFrame,
    *,
    materia: str = "mate",
    col_rbd: str = "rbd",
    solo_muestra_analitica: bool = False,
) -> pd.DataFrame:
    """Arma la tabla analítica de una fila por establecimiento.

    Parameters
    ----------
    resultados : pandas.DataFrame
        Salida de :func:`resultado.construir_resultados` +
        :func:`resultado.clasificar_cambio_nivel`.
    ancho : pandas.DataFrame
        Panel ancho (completo o reducido).
    materia : str, optional
        Materia de la que se toma la matrícula.
    col_rbd : str, optional
    solo_muestra_analitica : bool, optional
        Si ``True`` conserva únicamente los establecimientos con ambas
        medidas de resultado válidas. Por defecto ``False``: se conservan
        todos, y el filtrado se hace explícitamente en el análisis, de modo
        que el sesgo de selección pueda cuantificarse.

    Returns
    -------
    pandas.DataFrame
        Tabla analítica con resultados, covariables y variables geográficas.
    """
    cols = [col_rbd] + [c for c in COVARIABLES_ANCHO if c in ancho.columns]
    base = resultados.merge(ancho[cols], on=col_rbd, how="left")

    geo = variables_geograficas(ancho)
    base = base.merge(geo, on=col_rbd, how="left")

    mat = matricula_periodo(ancho, materia=materia)
    if not mat.empty:
        base = base.merge(mat, on=col_rbd, how="left")

    if solo_muestra_analitica and "muestra_analitica" in base.columns:
        antes = len(base)
        base = base[base["muestra_analitica"]].copy()
        log.info("Filtrado a muestra analítica: %d de %d establecimientos.",
                 len(base), antes)

    log.info("Dataset analítico: %d filas x %d columnas.", *base.shape)
    return base


def resumen_por_grupo(
    datos: pd.DataFrame,
    covariable: str,
    medidas: tuple[str, ...] = ("caida_inicial", "recuperacion",
                                "recuperacion_relativa"),
    *,
    minimo: int = 20,
) -> pd.DataFrame:
    """Descriptivos de las medidas de resultado por categoría de una covariable.

    Parameters
    ----------
    datos : pandas.DataFrame
        Tabla analítica.
    covariable : str
        Columna categórica por la que agrupar.
    medidas : tuple[str, ...], optional
    minimo : int, optional
        Categorías con menos de este número de establecimientos se marcan,
        para no leer promedios de grupos diminutos como si fueran estables.

    Returns
    -------
    pandas.DataFrame
        Por categoría: n y, para cada medida, media y mediana.
    """
    presentes = [m for m in medidas if m in datos.columns]
    agg = {"n": (covariable, "size")}
    for m in presentes:
        agg[f"{m}_media"] = (m, "mean")
        agg[f"{m}_mediana"] = (m, "median")

    out = datos.groupby(covariable, dropna=False).agg(**agg).round(2)
    out["grupo_pequeno"] = out["n"] < minimo
    return out.sort_values("n", ascending=False)
