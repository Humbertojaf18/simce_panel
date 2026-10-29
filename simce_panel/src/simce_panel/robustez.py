"""
robustez.py
===========

Verificaciones de robustez solicitadas en la revisión del 1-oct-2026.

Cada función responde una observación concreta del profesor guía y devuelve
números que pueden citarse en el texto:

- :func:`confiabilidad_pre`            : comentario 37 (regresión a la media).
- :func:`correlacion_por_construccion` : comentario 11 (puntaje 2022 compartido).
- :func:`traspasos_slep`               : comentario 14 (potencia de H5).
- :func:`sensibilidad_moran`           : comentarios 10, 26 y 39 (matriz de
  pesos y Moran mínimo detectable).
- :func:`moran_con_contraccion`        : comentario 39 (atenuación por ruido).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Desviación estándar individual de la escala SIMCE.
DE_INDIVIDUAL: float = 50.0


def confiabilidad_pre(ancho: pd.DataFrame, resultados: pd.DataFrame, *,
                      materia: str = "mate", col_rbd: str = "rbd") -> dict:
    """Contrasta la regresión a la media con la confiabilidad del PRE.

    Dos correlaciones de signo opuesto ---r(PRE, caída) negativa y
    r(PRE, puntaje 2022) positiva--- no prueban regresión a la media: son
    compatibles con un efecto real del nivel previo. La prueba correcta
    compara la correlación observada con la que produciría el puro error de
    medición del PRE. Bajo error clásico, cov(PRE, caída) = -var(error), y la
    varianza del error se obtiene de la confiabilidad, estimada por mitades
    (2015-16 frente a 2017-18) con la corrección de Spearman-Brown.
    """
    c = lambda t: pd.to_numeric(ancho[f"prom_{materia}2m_rbd_{t}"], errors="coerce")
    h1 = (c(2015) + c(2016)) / 2
    h2 = (c(2017) + c(2018)) / 2
    m = resultados.set_index(col_rbd)
    ok = h1.notna() & h2.notna() & ancho[col_rbd].isin(m.index)
    r_half = float(np.corrcoef(h1[ok], h2[ok])[0, 1])
    rel = 2 * r_half / (1 + r_half)
    var_err = (1 - rel) * m["puntaje_pre"].var()
    r_esp = -var_err / (m["puntaje_pre"].std() * m["caida_inicial"].std())
    r_obs = m["puntaje_pre"].corr(m["caida_inicial"])
    out = {"r_mitades": round(r_half, 3), "confiabilidad": round(rel, 3),
           "r_observada": round(float(r_obs), 3),
           "r_esperada_solo_error": round(float(r_esp), 3)}
    log.info("Confiabilidad PRE: %s", out)
    return out


def correlacion_por_construccion(resultados: pd.DataFrame,
                                 n_2022: pd.Series, *,
                                 col_rbd: str = "rbd") -> dict:
    """Parte de la correlación caída-recuperación que se debe al puntaje 2022.

    El puntaje de 2022 es el punto final de la caída y el primer punto de la
    pendiente. Su error de muestreo e entra con signo + en la caída y con peso
    -0,3 en la pendiente de mínimos cuadrados sobre t = 0..3, de modo que
    produce cov = -0,3 var(e) por construcción. var(e) se aproxima por
    DE_individual^2 / n evaluados en 2022.
    """
    m = resultados.copy()
    m["n22"] = m[col_rbd].map(n_2022)
    var_e = float((DE_INDIVIDUAL ** 2 / m["n22"]).median())
    r_obs = m["caida_inicial"].corr(m["recuperacion"])
    r_esp = -0.3 * var_e / (m["caida_inicial"].std() * m["recuperacion"].std())
    return {"r_observada": round(float(r_obs), 3),
            "r_esperada_por_ruido_2022": round(float(r_esp), 3),
            "var_ruido_2022": round(var_e, 1)}


def traspasos_slep(largo: pd.DataFrame, rbd_muestra, *,
                   col_depe: str = "dependencia", col_rbd: str = "rbd",
                   col_anio: str = "año") -> dict:
    """Cuenta traspasos efectivos de dependencia municipal a SLEP (comentario 14).

    Compara la dependencia más frecuente del período previo (2015-2018) con la
    del posterior (2022-2025). Un cambio de un año aislado no cuenta como
    traspaso.
    """
    def moda(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else None
    d = largo[largo[col_rbd].isin(set(rbd_muestra))]
    pre = d[d[col_anio].between(2015, 2018)].groupby(col_rbd)[col_depe].agg(moda)
    post = d[d[col_anio].between(2022, 2025)].groupby(col_rbd)[col_depe].agg(moda)
    t = pd.DataFrame({"pre": pre, "post": post}).dropna()
    muni = {"Municipal DAEM", "Municipal Corporación"}
    slep = "Servicio Local de Educación"
    return {"cambian_dependencia": int((t.pre != t.post).sum()),
            "traspaso_municipal_a_slep": int((t.pre.isin(muni) & (t.post == slep)).sum()),
            "municipales_sin_cambio": int((t.pre.isin(muni) & t.post.isin(muni)).sum())}


def _coords_km(df, col_lat="latitud", col_lon="longitud"):
    R = 6371.0
    lat0 = np.radians(df[col_lat].mean())
    return np.column_stack([R * np.radians(df[col_lon]) * np.cos(lat0),
                            R * np.radians(df[col_lat])])


def sensibilidad_moran(df: pd.DataFrame, columna: str, *,
                       ks: tuple[int, ...] = (4, 6, 8, 12, 20),
                       radios_km: tuple[float, ...] = (5.0, 10.0),
                       permutaciones: int = 999, semilla: int | None = 2026,
                       distancia_minima_km: float = 0.05) -> pd.DataFrame:
    """Moran bajo distintas matrices de pesos y Moran mínimo detectable.

    El Moran mínimo detectable (MDE) es el valor que la prueba detectaría con
    potencia del 80 % a un nivel unilateral del 5 %, aproximado como
    (1,645 + 0,842) veces la desviación estándar de la distribución de
    permutación. Si el Moran observado es comparable al MDE y la prueba lo
    detecta, el resultado es "detectable pero pequeño", no "falta de
    potencia".
    """
    from esda.moran import Moran
    from libpysal.weights import KNN, DistanceBand

    d = df.dropna(subset=[columna, "latitud", "longitud"])
    xy = _coords_km(d)
    # Escuelas con coordenadas idénticas (3 en la muestra analítica): con
    # distancia inversa, la distancia 0 no se almacena y esos pares quedaban
    # fuera de la vecindad. Se separan 50 m para que cuenten como vecinas; el
    # efecto en el Moran es de una diezmilésima.
    _, ix = np.unique(np.round(xy, 6), axis=0, return_inverse=True)
    rep = pd.Series(ix).duplicated(keep="first").to_numpy()
    if rep.any():
        xy = xy.copy(); xy[rep, 0] += distancia_minima_km
    y = d[columna].to_numpy(dtype="float64")
    pesos = [(f"kNN k={k}", KNN.from_array(xy, k=k)) for k in ks]
    # Algunas versiones de scipy elevan a -1 la diagonal de ceros de la matriz
    # de distancias antes de descartarla y emiten "divide by zero": el aviso es
    # inocuo (los pesos finales son finitos) y se silencia.
    with np.errstate(divide="ignore"):
        pesos += [(f"distancia inversa <= {r:g} km",
                   DistanceBand.from_array(xy, threshold=r, binary=False, alpha=-1.0,
                                           silence_warnings=True)) for r in radios_km]
    filas = []
    for nombre, w in pesos:
        w.transform = "r"
        if semilla is not None:
            np.random.seed(semilla)
        mi = Moran(y, w, permutations=permutaciones)
        filas.append({"pesos": nombre, "I": round(float(mi.I), 4),
                      "p": round(float(mi.p_sim), 3),
                      "MDE": round(float((1.645 + 0.842) * np.std(mi.sim)), 4)})
    return pd.DataFrame(filas)


def moran_con_contraccion(df: pd.DataFrame, residuo: str, var_error: pd.Series,
                          *, k: int = 8, permutaciones: int = 999) -> dict:
    """Moran del residuo tras contracción empírica Bayes (comentario 39).

    Cada residuo se multiplica por su confiabilidad
    lambda_i = tau^2 / (tau^2 + var_error_i), donde tau^2 es la varianza
    verdadera estimada. El ruido de medición atenúa el Moran observado; bajo
    error clásico, el Moran verdadero es aproximadamente el observado dividido
    por la confiabilidad media.
    """
    from esda.moran import Moran
    from libpysal.weights import KNN

    d = df.dropna(subset=[residuo, "latitud", "longitud"]).copy()
    ve = var_error.reindex(d.index).fillna(var_error.median())
    tau2 = max(d[residuo].var() - ve.mean(), 1e-6)
    lam = tau2 / (tau2 + ve)
    w = KNN.from_array(_coords_km(d), k=k); w.transform = "r"
    # Corrección del 24-oct-2026: la contracción es hacia la media del residuo.
    # Multiplicar el residuo sin centrar por lambda introducía una variación
    # artificial ligada al tamaño de la escuela (residuos con media 103 en la
    # versión anterior) e inflaba el Moran contraído de 0,036 a 0,063.
    r = d[residuo] - d[residuo].mean()
    np.random.seed(2026)
    i_obs = Moran(r.to_numpy(float), w, permutations=permutaciones)
    np.random.seed(2026)
    i_eb = Moran((lam * r).to_numpy(float), w, permutations=permutaciones)
    return {"confiabilidad_media": round(float(lam.mean()), 3),
            "I_observado": round(float(i_obs.I), 4),
            "I_contraido": round(float(i_eb.I), 4), "p_contraido": float(i_eb.p_sim),
            "I_corregido_atenuacion": round(float(i_obs.I / lam.mean()), 4)}
