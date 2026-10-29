"""
prediccion.py
=============

Validación predictiva fuera de muestra (criterio de éxito C3; rúbrica,
criterio 4: "línea base implementada").

Diseño
------
Se reserva el último año (2025) y se entrena con 2015-2024. Cada modelo predice
el puntaje 2025 de cada establecimiento y se compara con el observado. Es una
validación **temporal por bloque**: el año de prueba no participa del ajuste,
lo que evita la fuga de información propia de una partición aleatoria de
observaciones de un mismo establecimiento.

Modelos comparados (de menor a mayor complejidad)
-------------------------------------------------
- ``M0 media global``      : promedio nacional de 2022-2024, igual para todos.
- ``M1 persistencia``      : el puntaje 2024 de cada establecimiento.
- ``M2 tendencia propia``  : recta de mínimos cuadrados 2022-2024 de cada
  establecimiento, extrapolada a 2025.
- ``M3 lineal jerárquico`` : modelo mixto frecuentista con intercepto y
  pendiente aleatorios por establecimiento.
- ``M4 bayesiano``         : modelo de medición de :mod:`bayes`, que además
  entrega un intervalo predictivo por establecimiento.

Lo que se espera
----------------
M2 extrapola tres puntos ruidosos y debería ser el peor de los modelos por
establecimiento: es el caso que motiva el agrupamiento parcial. M3 y M4 deben
mejorar sobre M2; la pregunta de C3 es si M4 mejora sobre M3.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ANIO_PRUEBA: int = 2025


def _metricas(y: np.ndarray, yhat: np.ndarray) -> dict:
    e = yhat - y
    return {"RMSE": round(float(np.sqrt(np.mean(e ** 2))), 2),
            "MAE": round(float(np.mean(np.abs(e))), 2),
            "sesgo": round(float(np.mean(e)), 2)}


def preparar_particion(largo: pd.DataFrame, *, col_rbd: str = "rbd", col_anio: str = "anio",
                       col_y: str = "y", min_puntaje: float = 100.0,
                       anio_prueba: int = ANIO_PRUEBA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento (< 2025) y prueba (2025).

    Solo se evalúan establecimientos con puntaje en 2025 y con al menos dos
    años en 2022-2024, de modo que todos los modelos puedan predecirlos y la
    comparación sea sobre el mismo conjunto.
    """
    d = largo[largo[col_y] >= min_puntaje].copy()
    ent = d[d[col_anio] < anio_prueba].copy()
    pru = d[d[col_anio] == anio_prueba].copy()
    n_post = ent[ent[col_anio] >= 2022].groupby(col_rbd)[col_anio].nunique()
    elegibles = set(n_post[n_post >= 2].index) & set(pru[col_rbd])
    pru = pru[pru[col_rbd].isin(elegibles)]
    log.info("Entrenamiento: %d obs.; prueba: %d establecimientos en %d.",
             len(ent), len(pru), anio_prueba)
    return ent, pru


def predicciones_referencia(ent: pd.DataFrame, pru: pd.DataFrame, *, col_rbd: str = "rbd",
                            col_anio: str = "anio", col_y: str = "y") -> pd.DataFrame:
    """Predicciones de M0, M1 y M2 para cada establecimiento de prueba."""
    post = ent[ent[col_anio] >= 2022]
    out = pru[[col_rbd, col_y]].rename(columns={col_y: "observado"}).set_index(col_rbd)
    out["M0 media global"] = post[col_y].mean()
    ultimo = post.sort_values(col_anio).groupby(col_rbd).last()
    out["M1 persistencia"] = ultimo[col_y]

    def extrapolar(g):
        if g[col_anio].nunique() < 2:
            return g[col_y].iloc[-1]
        b, a = np.polyfit(g[col_anio] - 2022, g[col_y], 1)
        return a + b * (ANIO_PRUEBA - 2022)
    out["M2 tendencia propia"] = post.groupby(col_rbd).apply(extrapolar)
    return out


def prediccion_mixto(ent: pd.DataFrame, pru: pd.DataFrame, *, col_rbd: str = "rbd",
                     col_anio: str = "anio", col_y: str = "y",
                     caida_aleatoria: bool = False) -> pd.Series:
    """M3: modelo lineal jerárquico frecuentista (statsmodels MixedLM).

    Efectos fijos: nivel, salto en 2022 y pendiente posterior. Efectos
    aleatorios por establecimiento: intercepto y pendiente posterior y, con
    ``caida_aleatoria=True`` (M3b), también el salto de 2022. La predicción
    usa los efectos aleatorios estimados (BLUP).

    M3b es la línea base justa frente a M4: sin caída aleatoria, M3 obliga a
    todas las escuelas a caer lo mismo en 2022, y M4 le gana sobre todo por eso
    (verificación del 24-oct-2026: M3b 15,66 frente a M4 15,64 de RMSE).
    """
    import statsmodels.formula.api as smf

    d = ent.copy()
    d["post"] = (d[col_anio] >= 2022).astype(float)
    d["t_post"] = np.where(d["post"] == 1, d[col_anio] - 2022, 0.0)
    re_formula = "~post + t_post" if caida_aleatoria else "~t_post"
    modelo = smf.mixedlm(f"{col_y} ~ post + t_post", d, groups=d[col_rbd],
                         re_formula=re_formula)
    res = modelo.fit(method="lbfgs", maxiter=3000)
    if not np.isfinite(res.llf):
        res = modelo.fit(method="powell", maxiter=5000)
    fe = res.fe_params
    x = {"Group": 1.0, "Intercept": 1.0, "post": 1.0, "t_post": float(ANIO_PRUEBA - 2022)}
    pred = {}
    for rbd in pru[col_rbd]:
        valor = fe["Intercept"] + fe["post"] + fe["t_post"] * (ANIO_PRUEBA - 2022)
        re = res.random_effects.get(rbd)
        if re is not None:
            valor += sum(float(re[k]) * x.get(k, 0.0) for k in re.index)
        pred[rbd] = valor
    nombre = "M3b caída aleatoria" if caida_aleatoria else "M3 lineal jerárquico"
    return pd.Series(pred, name=nombre)


def prediccion_bayes(mcmc, datos: dict, pru: pd.DataFrame, n_2025: pd.Series, *,
                     col_rbd: str = "rbd", prob: float = 0.94) -> pd.DataFrame:
    """M4: predicción del modelo de medición para 2025, con intervalo predictivo.

    La media predictiva es a_i + d_i + 3 b_i (más el efecto del nivel previo si
    se estimó). El intervalo agrega el error de medición del año de prueba,
    sqrt(sigma_w^2 / n_2025 + sigma_e^2), de modo que es más ancho para los
    establecimientos con pocos evaluados.
    """
    from .bayes import ESCALA_SIMCE
    s = mcmc.get_samples()
    a = np.asarray(s["a"]); d = np.asarray(s["caida"]) / ESCALA_SIMCE
    b = np.asarray(s["recuperacion"]) / ESCALA_SIMCE
    mu = a + d + b * (ANIO_PRUEBA - 2022)
    idx = {r: i for i, r in enumerate(datos["rbd"])}
    rbds = [r for r in pru[col_rbd] if r in idx]
    cols = [idx[r] for r in rbds]
    mu = mu[:, cols]
    n = n_2025.reindex(rbds).fillna(n_2025.median()).to_numpy()
    sd = np.sqrt(np.asarray(s["s_w"])[:, None] ** 2 / n[None, :]
                 + np.asarray(s["s_e"])[:, None] ** 2)
    rng = np.random.default_rng(0)
    ypred = (mu + sd * rng.standard_normal(mu.shape)) * ESCALA_SIMCE + datos["media_y"]
    q = (1 - prob) / 2 * 100
    return pd.DataFrame({"M4 bayesiano": mu.mean(0) * ESCALA_SIMCE + datos["media_y"],
                         "M4_inf": np.percentile(ypred, q, 0),
                         "M4_sup": np.percentile(ypred, 100 - q, 0)}, index=rbds)


def comparar(tabla: pd.DataFrame, *, modelos: list[str] | None = None,
             col_obs: str = "observado") -> pd.DataFrame:
    """Métricas de cada modelo sobre el mismo conjunto de prueba.

    Incluye la mejora porcentual del RMSE respecto de la media global (M0) y
    del lineal jerárquico (M3), que es la comparación que exige el criterio C3.
    """
    modelos = modelos or [c for c in tabla.columns if c.startswith("M") and "_" not in c]
    t = tabla.dropna(subset=modelos + [col_obs])
    filas = [{"modelo": m, **_metricas(t[col_obs].to_numpy(), t[m].to_numpy())} for m in modelos]
    out = pd.DataFrame(filas).set_index("modelo")
    for ref in ("M0 media global", "M3 lineal jerárquico"):
        if ref in out.index:
            out[f"mejora_vs_{ref.split()[0]}_%"] = (
                100 * (1 - out["RMSE"] / out.loc[ref, "RMSE"])).round(1)
    out.attrs["n"] = len(t)
    return out


def diferencia_rmse_bootstrap(tabla: pd.DataFrame, referencia: str, candidato: str, *,
                              col_obs: str = "observado", n_boot: int = 2000,
                              semilla: int = 0) -> dict:
    """Intervalo bootstrap de RMSE(referencia) - RMSE(candidato).

    Con un único año de prueba, la comparación entre modelos necesita su propia
    medida de incertidumbre. Se remuestrean establecimientos (pareados: ambos
    modelos se evalúan sobre la misma réplica). Un intervalo que excluye el cero
    indica que la mejora no es atribuible al azar de la muestra de prueba.
    """
    t = tabla.dropna(subset=[referencia, candidato, col_obs])
    e_r = ((t[referencia] - t[col_obs]) ** 2).to_numpy()
    e_c = ((t[candidato] - t[col_obs]) ** 2).to_numpy()
    rng = np.random.default_rng(semilla)
    n = len(t)
    dif = np.array([np.sqrt(e_r[s].mean()) - np.sqrt(e_c[s].mean())
                    for s in (rng.integers(0, n, n) for _ in range(n_boot))])
    return {"diferencia_RMSE": round(float(np.sqrt(e_r.mean()) - np.sqrt(e_c.mean())), 2),
            "IC95": [round(float(np.percentile(dif, 2.5)), 2),
                     round(float(np.percentile(dif, 97.5)), 2)],
            "prob_candidato_mejor": round(float((dif > 0).mean()), 3),
            "pct_establecimientos_mejor": round(float((e_c < e_r).mean() * 100), 1),
            "n": n}
