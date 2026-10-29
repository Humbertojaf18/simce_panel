"""
contrastes.py
=============

Contrastes pendientes de la revisión del 1-oct-2026.

- **H5** (comentario 14): ¿el traspaso de un establecimiento municipal a un
  Servicio Local de Educación Pública (SLEP) se asocia a una recuperación
  distinta? Comparación emparejada por grupo socioeconómico y puntaje previo.
- **H3, componente entre asignaturas** (comentario 12): ¿el gradiente
  socioeconómico de la recuperación es mayor en Matemática que en Lectura?
  Contribución SHAP del GSE con intervalos bootstrap, y de su diferencia.

Los valores SHAP se obtienen con ``pred_contribs`` de xgboost, que implementa
el algoritmo exacto TreeSHAP (Lundberg y Lee, 2017) sin requerir la librería
``shap``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MUNICIPAL = {"Municipal DAEM", "Municipal Corporación"}
SLEP = "Servicio Local de Educación"
ORDEN_GSE = ("Bajo", "Medio bajo", "Medio", "Medio alto", "Alto")


# --------------------------------------------------------------------------- #
# H5: traspaso a SLEP
# --------------------------------------------------------------------------- #

def clasificar_traspasos(largo: pd.DataFrame, *, col_depe: str = "cod_depe1_cat",
                         col_rbd: str = "rbd", col_anio: str = "año") -> pd.DataFrame:
    """Clasifica cada establecimiento como traspaso, control u otro.

    Traspaso: dependencia modal municipal en 2015-2018 y SLEP en 2022-2025.
    Control: municipal en ambos períodos. El resto no participa del contraste.
    """
    def moda(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else None

    pre = largo[largo[col_anio].between(2015, 2018)].groupby(col_rbd)[col_depe].agg(moda)
    post = largo[largo[col_anio].between(2022, 2025)].groupby(col_rbd)[col_depe].agg(moda)
    t = pd.DataFrame({"depe_pre": pre, "depe_post": post}).dropna()
    t["grupo_h5"] = np.where(t.depe_pre.isin(MUNICIPAL) & (t.depe_post == SLEP), "traspaso",
                    np.where(t.depe_pre.isin(MUNICIPAL) & t.depe_post.isin(MUNICIPAL),
                             "control", "otro"))
    return t.reset_index().rename(columns={"index": col_rbd})


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Diferencia de medias estandarizada (balance de covariables)."""
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def h5_emparejado(datos: pd.DataFrame, resultado: str, *, col_grupo: str = "grupo_h5",
                  col_gse: str = "cod_grupo_cat", col_pre: str = "puntaje_pre",
                  caliper_de: float = 0.25, n_boot: int = 2000, semilla: int = 0) -> dict:
    """Efecto del traspaso a SLEP sobre los tratados (ATT), por emparejamiento.

    Cada establecimiento traspasado se empareja con el control de su mismo GSE
    más cercano en puntaje previo, con reemplazo y dentro de un caliper de
    ``caliper_de`` desviaciones estándar del puntaje previo. El intervalo se
    obtiene por bootstrap de los pares. Se reporta además la diferencia mínima
    detectable con los tamaños de muestra disponibles.
    """
    d = datos.dropna(subset=[resultado, col_gse, col_pre, col_grupo])
    tr = d[d[col_grupo] == "traspaso"]; co = d[d[col_grupo] == "control"]
    caliper = caliper_de * d[col_pre].std()

    pares = []
    for _, fila in tr.iterrows():
        cand = co[co[col_gse] == fila[col_gse]]
        if cand.empty:
            continue
        dist = (cand[col_pre] - fila[col_pre]).abs()
        j = dist.idxmin()
        if dist[j] <= caliper:
            pares.append((fila[resultado], cand.loc[j, resultado],
                          fila[col_pre], cand.loc[j, col_pre]))
    if not pares:
        raise ValueError("Ningún tratado encontró control dentro del caliper.")
    P = np.array(pares)
    dif = P[:, 0] - P[:, 1]
    rng = np.random.default_rng(semilla)
    boot = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(n_boot)]

    # Corrección del 24-oct-2026: el MDE se calcula con el error estándar del
    # propio estimador emparejado (bootstrap de los pares). La fórmula anterior
    # suponía dos muestras independientes de 115 y 712 escuelas, pero el ATT
    # compara 113 pares, y daba un MDE demasiado optimista.
    ee = float(np.std(boot, ddof=1))
    mde = 2.8 * ee
    return {
        "resultado": resultado,
        "n_traspaso": len(tr), "n_control": len(co), "n_emparejados": len(P),
        "ATT": round(float(dif.mean()), 3),
        "IC95": [round(float(np.percentile(boot, 2.5)), 3),
                 round(float(np.percentile(boot, 97.5)), 3)],
        "SMD_pre_antes": round(_smd(tr[col_pre], co[col_pre]), 3),
        "SMD_pre_despues": round(_smd(pd.Series(P[:, 2]), pd.Series(P[:, 3])), 3),
        "EE_bootstrap": round(ee, 3),
        "MDE_80pct": round(float(mde), 3),
    }


# --------------------------------------------------------------------------- #
# H3 entre asignaturas: gradiente SHAP del GSE con bootstrap
# --------------------------------------------------------------------------- #

def preparar_matriz(datos: pd.DataFrame, numericas: list[str], *,
                    col_gse: str = "cod_grupo_cat",
                    categoricas: tuple[str, ...] = ("cod_depe1_cat", "cod_rural_rbd_cat")) -> pd.DataFrame:
    """Matriz de predictores con el GSE como variable ordinal (0 = Bajo).

    El GSE se codifica como una sola variable ordinal para que su contribución
    SHAP sea un único número por establecimiento, directamente comparable entre
    modelos.
    """
    X = pd.DataFrame(index=datos.index)
    X["gse_ord"] = datos[col_gse].map({g: i for i, g in enumerate(ORDEN_GSE)})
    for c in numericas:
        if c in datos.columns:
            X[c] = pd.to_numeric(datos[c], errors="coerce")
    for c in categoricas:
        if c in datos.columns:
            X = X.join(pd.get_dummies(datos[c], prefix=c, dtype=float))
    return X


def _gradiente(modelo, X: pd.DataFrame) -> float:
    """Contribución SHAP media del GSE en Alto menos la de Bajo (puntos/año)."""
    import xgboost as xgb
    contrib = modelo.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    phi = contrib[:, list(X.columns).index("gse_ord")]
    g = X["gse_ord"].to_numpy()
    return float(phi[g == 4].mean() - phi[g == 0].mean())


def _modelo(semilla: int):
    import xgboost as xgb
    # Configuración deliberadamente conservadora: el objetivo es atribuir, no
    # predecir. Árboles poco profundos y submuestreo reducen el sobreajuste a
    # establecimientos individuales.
    return xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                            random_state=semilla, n_jobs=2)


def gradiente_shap_bootstrap(conjuntos: dict[str, tuple[pd.DataFrame, pd.Series]], *,
                             n_boot: int = 200, semilla: int = 0) -> dict:
    """Gradiente SHAP del GSE por asignatura y su diferencia, con bootstrap conjunto.

    ``conjuntos`` asocia cada asignatura a ``(X, y)`` con índice común (RBD).
    En cada réplica se remuestrean los mismos establecimientos para todas las
    asignaturas y se reajustan los modelos, de modo que el intervalo de la
    diferencia incorpora la correlación entre ambas estimaciones.
    """
    comunes = None
    for X, y in conjuntos.values():
        idx = X.dropna().index.intersection(y.dropna().index)
        comunes = idx if comunes is None else comunes.intersection(idx)
    comunes = comunes.sort_values()
    nombres = list(conjuntos)

    def estimar(muestra, rs):
        out = {}
        for k, (X, y) in conjuntos.items():
            Xs, ys = X.loc[muestra], y.loc[muestra]
            m = _modelo(rs).fit(Xs, ys)
            out[k] = _gradiente(m, X.loc[comunes])
        return out

    puntual = estimar(comunes, semilla)
    rng = np.random.default_rng(semilla)
    reps = {k: [] for k in nombres}
    for b in range(n_boot):
        muestra = comunes[rng.integers(0, len(comunes), len(comunes))]
        est = estimar(muestra, semilla + b + 1)
        for k in nombres:
            reps[k].append(est[k])

    ic = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    res = {"n_establecimientos": len(comunes), "n_boot": n_boot}
    for k in nombres:
        res[k] = {"gradiente": round(puntual[k], 3), "IC95": ic(reps[k])}
    if len(nombres) == 2:
        a, b = nombres
        dif = np.array(reps[a]) - np.array(reps[b])
        res[f"diferencia_{a}_menos_{b}"] = {"estimacion": round(puntual[a] - puntual[b], 3),
                                            "IC95": ic(dif),
                                            "prob_mayor_que_0": round(float((dif > 0).mean()), 3)}
    return res


def r2_por_bloques(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, *,
                   n_folds: int = 5, semilla: int = 0) -> float:
    """R² fuera de muestra con validación cruzada por comunas (sin fuga espacial)."""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    idx = X.dropna().index.intersection(y.dropna().index).intersection(grupos.dropna().index)
    X, y, g = X.loc[idx], y.loc[idx], grupos.loc[idx]
    pred = pd.Series(index=idx, dtype=float)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, g):
        m = _modelo(semilla).fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[te] = m.predict(X.iloc[te])
    return round(float(r2_score(y, pred)), 3)
