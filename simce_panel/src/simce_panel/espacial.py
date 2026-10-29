"""
espacial.py
===========

Análisis espacial de las variables de resultado (OE3).

La pregunta que responde
------------------------
Detectar autocorrelación espacial positiva en un indicador escolar chileno no
constituye, por sí solo, un hallazgo. El sistema escolar del país presenta uno
de los niveles de segregación socioeconómica más altos de la OCDE, de modo que
la composición social de cada establecimiento está fuertemente determinada por
su ubicación residencial. Bajo esas condiciones, casi cualquier variable
correlacionada con el nivel socioeconómico exhibirá un índice de Moran positivo
y significativo por construcción.

La pregunta informativa es otra: **¿queda estructura espacial después de
descontar la composición?** Por eso el contraste central se aplica sobre los
residuos de un modelo que ya controla grupo socioeconómico, dependencia
administrativa y puntaje pre-pandemia, y no sobre la variable cruda.

Funciones principales
---------------------
- :func:`construir_pesos`        : matriz de pesos por k vecinos más cercanos.
- :func:`moran_global`           : índice de Moran con inferencia por permutaciones.
- :func:`sensibilidad_k`         : Moran para distintos valores de k.
- :func:`modelo_base`            : modelo lineal jerárquico (línea base) y residuos.
- :func:`lisa_con_fdr`           : LISA con control de la tasa de falso descubrimiento.
- :func:`comparar_conglomerados` : contingencia entre dos clasificaciones LISA.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Número de vecinos por defecto para la matriz de pesos.
K_VECINOS_DEFECTO: int = 8

#: Permutaciones para la inferencia. Con 9.999 el p-valor mínimo alcanzable
#: es 0,0001, suficiente para sobrevivir a la corrección por comparaciones
#: múltiples sobre unos 2.800 contrastes locales.
PERMUTACIONES: int = 9999

#: Semilla de las permutaciones. Sin ella, los valores p y los conteos de LISA
#: cambian levemente en cada ejecución y el informe no coincide con el cuaderno.
SEMILLA: int = 2026

#: Etiquetas de los cuadrantes LISA.
ETIQUETAS_LISA: dict[int, str] = {
    0: "no significativo",
    1: "Alto-Alto",
    2: "Bajo-Alto",
    3: "Bajo-Bajo",
    4: "Alto-Bajo",
}


def construir_pesos(
    df: pd.DataFrame,
    *,
    k: int = K_VECINOS_DEFECTO,
    col_lat: str = "latitud",
    col_lon: str = "longitud",
):
    """Construye la matriz de pesos espaciales por k vecinos más cercanos.

    Se usa k vecinos y no contigüidad porque los establecimientos son puntos,
    no polígonos, y no comparten fronteras. Los pesos se estandarizan por fila,
    de modo que el rezago espacial de cada unidad sea el promedio de sus
    vecinos y no una suma que dependa de cuántos tenga.

    Las coordenadas se proyectan a un plano local antes de calcular distancias:
    en Chile un grado de longitud mide unos 104 km en Arica y 62 km en Punta
    Arenas, de modo que usar grados como si fueran un plano haría que el
    vecindario de un colegio austral fuera geográficamente más ancho que el de
    uno del norte, sesgando el análisis justo en la dimensión estudiada.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con coordenadas, sin nulos en ellas.
    k : int, optional
        Número de vecinos.
    col_lat, col_lon : str, optional

    Returns
    -------
    libpysal.weights.W
        Matriz de pesos estandarizada por filas.
    """
    from libpysal.weights import KNN

    coords = _proyectar_km(df[col_lat].to_numpy(dtype="float64"),
                           df[col_lon].to_numpy(dtype="float64"))
    w = KNN.from_array(coords, k=k)
    w.transform = "r"
    log.info("Pesos espaciales: %d unidades, k=%d.", w.n, k)
    return w


def _proyectar_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Proyección equirectangular local a kilómetros.

    Suficiente para definir vecindarios: preserva distancias con error menor
    al 1 % en radios de pocas decenas de kilómetros, que es la escala en la
    que operan los k vecinos más cercanos.
    """
    R = 6371.0
    lat0 = np.radians(np.nanmean(lat))
    x = R * np.radians(lon) * np.cos(lat0)
    y = R * np.radians(lat)
    return np.column_stack([x, y])


def moran_global(
    valores: pd.Series | np.ndarray,
    w,
    *,
    permutaciones: int = PERMUTACIONES,
    etiqueta: str = "",
    semilla: int | None = SEMILLA,
) -> dict[str, float | str]:
    """Índice de Moran global con inferencia por permutaciones.

    Parameters
    ----------
    valores : pandas.Series | numpy.ndarray
        Variable a evaluar; no debe contener nulos.
    w : libpysal.weights.W
    permutaciones : int, optional
    etiqueta : str, optional
        Nombre de la variable, para el reporte.

    Returns
    -------
    dict
        ``variable``, ``n``, ``I``, ``E[I]``, ``z``, ``p_sim``.
    """
    from esda.moran import Moran

    y = np.asarray(valores, dtype="float64")
    if semilla is not None:
        np.random.seed(semilla)
    mi = Moran(y, w, permutations=permutaciones)
    return {
        "variable": etiqueta or "valores",
        "n": int(len(y)),
        "I": round(float(mi.I), 4),
        "E[I]": round(float(mi.EI), 4),
        "z": round(float(mi.z_sim), 2),
        "p_sim": float(mi.p_sim),
    }


def sensibilidad_k(
    df: pd.DataFrame,
    columna: str,
    *,
    ks: tuple[int, ...] = (4, 6, 8, 12, 20),
    col_lat: str = "latitud",
    col_lon: str = "longitud",
    permutaciones: int = 999,
) -> pd.DataFrame:
    """Evalúa cómo cambia el índice de Moran según el número de vecinos.

    La elección de ``k`` es arbitraria y puede condicionar el resultado. Este
    reporte documenta que la conclusión no depende de un valor particular.

    Returns
    -------
    pandas.DataFrame
        Una fila por ``k``.
    """
    d = df.dropna(subset=[columna, col_lat, col_lon])
    filas = []
    for k in ks:
        w = construir_pesos(d, k=k, col_lat=col_lat, col_lon=col_lon)
        r = moran_global(d[columna], w, permutaciones=permutaciones,
                         etiqueta=columna)
        r["k"] = k
        filas.append(r)
    return pd.DataFrame(filas)[["k", "variable", "n", "I", "z", "p_sim"]]


#: Covariables por defecto de la línea base. Incluyen la composición
#: socioeconómica, la dependencia, la ruralidad y el nivel previo, que son
#: justamente los factores que podrían generar la autocorrelación observada.
COVARIABLES_BASE: tuple[str, ...] = (
    "cod_grupo_cat", "cod_depe1_cat", "cod_rural_rbd_cat", "puntaje_pre",
)


def modelo_base(
    df: pd.DataFrame,
    dependiente: str,
    *,
    covariables: tuple[str, ...] = COVARIABLES_BASE,
    grupo: str = "comuna_dir",
    incluir_ruptura: bool = True,
):
    """Ajusta el modelo lineal jerárquico que sirve de línea base.

    La estructura jerárquica ---intercepto aleatorio por comuna--- reconoce
    que los establecimientos están anidados en territorios y que dos colegios
    de una misma comuna comparten condiciones no observadas. Sus residuos son
    el insumo del contraste de H1: si tras descontar composición y territorio
    administrativo aún queda autocorrelación espacial, esa estructura no se
    explica por las covariables incluidas.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla analítica.
    dependiente : str
        Variable de resultado.
    covariables : tuple[str, ...], optional
    grupo : str, optional
        Variable de agrupamiento para el efecto aleatorio.
    incluir_ruptura : bool, optional
        Si ``True`` incorpora el indicador de ruptura sostenida como control.

    Returns
    -------
    tuple
        ``(resultado_statsmodels, df_usado)``. ``df_usado`` incluye la columna
        ``residuo`` y solo las filas efectivamente ajustadas.
    """
    import statsmodels.formula.api as smf

    usadas = [c for c in covariables if c in df.columns]
    if incluir_ruptura and "ruptura_sostenida" in df.columns:
        usadas = list(usadas) + ["ruptura_sostenida"]

    cols = [dependiente, grupo] + list(usadas)
    # Se arrastra el identificador del establecimiento para poder unir los
    # residuos con el resto de las tablas sin depender del índice.
    identificadores = [c for c in ("rbd",) if c in df.columns]
    d = df[identificadores + cols + ["latitud", "longitud"]].dropna().copy()

    if "ruptura_sostenida" in d.columns:
        d["ruptura_sostenida"] = d["ruptura_sostenida"].astype(int)

    terminos = []
    for c in usadas:
        if d[c].dtype == object or str(d[c].dtype).startswith("category"):
            terminos.append(f"C({c})")
        else:
            terminos.append(c)
    formula = f"{dependiente} ~ " + " + ".join(terminos)

    log.info("Modelo base: %s | agrupado por %s | n=%d", formula, grupo, len(d))
    modelo = smf.mixedlm(formula, data=d, groups=d[grupo])
    # Corrección del 24-oct-2026. Con L-BFGS el ajuste quedaba atascado en un
    # borde degenerado (log-verosimilitud infinita, intercepto fijo en 0 y
    # varianza comunal 0), lo que sesgaba los coeficientes y dejaba residuos
    # con media 103. Powell converge al óptimo real: varianza comunal pequeña
    # pero no nula. Se verifica que la log-verosimilitud sea finita.
    res = modelo.fit(method="powell", maxiter=5000)
    if not np.isfinite(res.llf):
        raise RuntimeError("El modelo base no convergió: log-verosimilitud no finita.")

    # Residuos marginales: se calculan solo con la parte fija del modelo.
    # Es necesario porque, cuando la varianza del efecto aleatorio colapsa a
    # cero (estructura singular), statsmodels no puede predecir los efectos
    # aleatorios y `resid` falla. Un efecto aleatorio nulo es en sí mismo un
    # resultado: significa que, una vez descontadas las covariables, no queda
    # varianza atribuible a la comuna.
    var_re = float(np.asarray(res.cov_re).ravel()[0]) if res.cov_re is not None else 0.0
    if var_re < 1e-8:
        log.warning("La varianza del efecto aleatorio por '%s' es "
                    "prácticamente nula (%.2e): tras controlar las "
                    "covariables no queda varianza comunal. Los residuos se "
                    "calculan solo con la parte fija.", grupo, var_re)
    d["residuo"] = (d[dependiente].to_numpy()
                    - np.asarray(res.predict(d), dtype="float64"))
    if abs(d["residuo"].mean()) > 0.05 * d["residuo"].std():
        log.warning("Los residuos tienen media %.2f: revisa el ajuste del modelo base.",
                    d["residuo"].mean())
    d.attrs["var_efecto_aleatorio"] = var_re
    r2_marginal = 1 - d["residuo"].var(ddof=0) / d[dependiente].var(ddof=0)
    log.info("Varianza de la dependiente explicada por el modelo: %.1f %%.",
             100 * r2_marginal)
    return res, d


def lisa_con_fdr(
    df: pd.DataFrame,
    columna: str,
    w,
    *,
    alfa: float = 0.05,
    permutaciones: int = PERMUTACIONES,
    semilla: int | None = SEMILLA,
) -> pd.DataFrame:
    """Calcula los indicadores locales (LISA) con control de falso descubrimiento.

    Con unas 2.800 unidades, LISA ejecuta un contraste por establecimiento. Sin
    corrección, a un nivel nominal de 0,05 cabría esperar unos 140
    establecimientos significativos por puro azar, cifra comparable a la de
    muchos conglomerados reales. Se aplica el procedimiento de
    Benjamini-Hochberg, que controla la proporción esperada de falsos
    positivos entre los resultados declarados significativos.

    Parameters
    ----------
    df : pandas.DataFrame
    columna : str
    w : libpysal.weights.W
    alfa : float, optional
    permutaciones : int, optional

    Returns
    -------
    pandas.DataFrame
        Columnas: ``I_local``, ``cuadrante``, ``p_sim``, ``p_fdr``,
        ``significativo``, ``conglomerado`` (etiqueta final, ya corregida).
    """
    from esda.moran import Moran_Local
    from statsmodels.stats.multitest import multipletests

    y = df[columna].to_numpy(dtype="float64")
    lisa = Moran_Local(y, w, permutations=permutaciones, seed=semilla)

    p = np.asarray(lisa.p_sim, dtype="float64")
    rechaza, p_fdr, _, _ = multipletests(p, alpha=alfa, method="fdr_bh")

    out = pd.DataFrame({
        "I_local": np.round(lisa.Is, 4),
        "cuadrante": lisa.q,
        "p_sim": p,
        "p_fdr": np.round(p_fdr, 4),
        "significativo": rechaza,
    }, index=df.index)

    out["conglomerado"] = np.where(
        out["significativo"],
        pd.Series(out["cuadrante"]).map(ETIQUETAS_LISA).to_numpy(),
        "no significativo")

    n_sin = int((p < alfa).sum())
    n_con = int(rechaza.sum())
    log.info("LISA sobre '%s': %d significativos sin corrección, %d tras FDR.",
             columna, n_sin, n_con)
    return out


def comparar_conglomerados(
    cong_a: pd.Series,
    cong_b: pd.Series,
    *,
    nombre_a: str = "A",
    nombre_b: str = "B",
) -> dict[str, object]:
    """Contrasta dos clasificaciones LISA (prueba de H2).

    H2 sostiene que los conglomerados de mayor caída inicial no coinciden
    espacialmente con los de menor recuperación posterior. Esta función tabula
    ambas clasificaciones y evalúa la independencia con una prueba chi-cuadrado.

    Parameters
    ----------
    cong_a, cong_b : pandas.Series
        Columnas ``conglomerado`` de dos ejecuciones de :func:`lisa_con_fdr`.
    nombre_a, nombre_b : str, optional

    Returns
    -------
    dict
        ``tabla`` (contingencia), ``chi2``, ``p``, ``gl`` y ``v_cramer``.
    """
    from scipy.stats import chi2_contingency

    tabla = pd.crosstab(cong_a.rename(nombre_a), cong_b.rename(nombre_b))
    chi2, p, gl, _ = chi2_contingency(tabla)
    n = tabla.to_numpy().sum()
    min_dim = min(tabla.shape) - 1
    v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else np.nan
    return {"tabla": tabla, "chi2": round(float(chi2), 2), "p": float(p),
            "gl": int(gl), "v_cramer": round(v, 3)}
