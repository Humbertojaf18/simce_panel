"""
bayes.py
========

Regresión lineal bayesiana jerárquica de **serie interrumpida** para la
trayectoria de cada establecimiento (sugerencia del profesor C. Pieringer).

Analogía con el ejemplo de referencia
-------------------------------------
El tutorial de NumPyro sobre fibrosis pulmonar modela, para cada paciente, la
evolución de la capacidad vital forzada a lo largo de las semanas, estimando
una recta por paciente con *agrupamiento parcial*: cada paciente tiene su
propio intercepto y pendiente, pero ambos se extraen de una distribución
común, de modo que los pacientes con pocas mediciones "toman prestada" fuerza
del conjunto. Aquí cada **establecimiento** es un paciente y cada **año** una
medición.

Adaptación: serie interrumpida
------------------------------
La recta simple del ejemplo no basta, porque la trayectoria escolar tiene un
quiebre: el hiato 2019-2021 y el golpe de la pandemia. Para cada
establecimiento ``i`` y año ``t`` se modela::

    y_it = a_i + d_i * POST_t + b_i * (t - 2022) * POST_t + e_it

- ``a_i`` : nivel pre-pandemia (2015-2018).
- ``d_i`` : **caída inicial**, el salto en 2022 respecto del nivel previo.
- ``b_i`` : **recuperación**, la pendiente anual desde 2022.

Las tres cantidades son las mismas variables de resultado de la tesis, pero
ahora se estiman **conjuntamente y con incertidumbre**, en lugar de calcularse
por separado con promedios y mínimos cuadrados escuela a escuela.

Qué gana la tesis con este modelo
---------------------------------
1. **Contracción hacia la media (shrinkage).** Las escuelas pequeñas o con
   años faltantes producen estimaciones ruidosas; el modelo las acerca a la
   media de su grupo en proporción a su incertidumbre. Resuelve el problema de
   inestabilidad de los establecimientos pequeños sin descartarlos.
2. **Incertidumbre por escuela.** Cada caída y cada pendiente tienen un
   intervalo de credibilidad, de modo que puede afirmarse cuándo la
   recuperación de una escuela es creíblemente distinta de cero.
3. **Efectos de grupo con intervalo.** Los coeficientes por grupo
   socioeconómico responden H3 con una medida de incertidumbre explícita.
4. **Explica el resultado espacial nulo.** Si pocas escuelas tienen una
   pendiente creíblemente distinta de cero, los contrastes locales (LISA) no
   pueden detectar conglomerados: el ruido individual domina. La estructura
   territorial debe modelarse en el nivel de grupo (comuna, región), no
   contrastarse escuela por escuela.

Modelo de medición (revisión del 1-oct-2026, comentarios 11 y 24)
------------------------------------------------------------------
El promedio SIMCE de un establecimiento es una estimación con error de
muestreo, mayor cuanto menos estudiantes rinden la prueba. Con la opción
``n_evaluados`` el modelo usa una varianza heterogénea por establecimiento-año:

    y_it ~ Normal(mu_it, sqrt(sigma_w^2 / n_it + sigma_e^2))

donde ``n_it`` es el número de evaluados, ``sigma_w`` la dispersión entre
estudiantes (cercana a 1 en la escala estandarizada, porque la escala SIMCE
tiene desviación individual de 50) y ``sigma_e`` la variación propia del
establecimiento entre años. Dos consecuencias:

- Los establecimientos pequeños aportan observaciones más ruidosas y su caída
  y pendiente se contraen más hacia la media de su grupo.
- La caída y la pendiente se estiman **conjuntamente**: el ruido del puntaje
  de 2022, que es a la vez punto final de una y punto de partida de la otra,
  queda absorbido por la verosimilitud y no se convierte en una correlación
  negativa espuria entre ambas.

Supuesto a vigilar
------------------
El modelo asume un nivel PRE **plano**. Si un grupo de escuelas venía en alza
antes de la pandemia, esa tendencia previa se confunde con "recuperación".
Por eso se incluye la opción ``tendencia_pre``, que añade una pendiente previa
y permite contrastar ese supuesto.

Funciones principales
---------------------
- :func:`preparar_datos`   : arma los arreglos del modelo desde el panel largo.
- :func:`modelo_serie`     : especificación NumPyro (parametrización no centrada).
- :func:`ajustar`          : muestreo NUTS con diagnósticos.
- :func:`resumen_grupos`   : efectos por grupo con intervalos de credibilidad.
- :func:`resumen_escuelas` : caída y recuperación por escuela, con intervalos.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Escala del SIMCE (desviación estándar de referencia). Se estandariza el
#: puntaje con ella para que los priors sean interpretables y el muestreador
#: trabaje en una escala cercana a la unidad.
ESCALA_SIMCE: float = 50.0

#: Año del quiebre: primera medición posterior al hiato.
ANIO_QUIEBRE: int = 2022

#: Orden del grupo socioeconómico, de menor a mayor.
ORDEN_GSE: tuple[str, ...] = ("Bajo", "Medio bajo", "Medio", "Medio alto", "Alto")


def preparar_datos(
    largo: pd.DataFrame,
    *,
    col_rbd: str = "rbd",
    col_anio: str = "anio",
    col_y: str = "y",
    col_grupo: str = "cod_grupo_cat",
    col_n: str | None = None,
    orden_grupo: tuple[str, ...] = ORDEN_GSE,
    min_anios: int = 5,
    min_puntaje: float = 100.0,
    exigir_2022: bool = True,
    covariables_escuela: pd.DataFrame | None = None,
) -> dict:
    """Convierte el panel largo en los arreglos que consume el modelo.

    Parameters
    ----------
    largo : pandas.DataFrame
        Una fila por establecimiento-año, con el puntaje y el grupo.
    col_rbd, col_anio, col_y, col_grupo : str, optional
    orden_grupo : tuple[str, ...], optional
        Categorías del grupo, en orden. Cada escuela se asigna a su categoría
        más frecuente en el período.
    min_anios : int, optional
        Años mínimos con puntaje por escuela. Con menos, la trayectoria no
        aporta información sobre la pendiente.
    min_puntaje : float, optional
        Descarta sentinelas de supresión codificados como puntaje.
    col_n : str | None, optional
        Columna con el número de evaluados. Si se indica, el modelo usa la
        varianza de medición heterogénea (modelo de medición).
    exigir_2022 : bool, optional
        Exige puntaje válido en el año del quiebre, del que depende la caída
        inicial (comentario 22 de la revisión).

    Returns
    -------
    dict
        Arreglos del modelo más las tablas de correspondencia (``rbd``,
        ``grupo``) para traducir los resultados.
    """
    d = largo[largo[col_y] >= min_puntaje].copy()
    n_anios = d.groupby(col_rbd)[col_anio].nunique()
    validas = n_anios[n_anios >= min_anios].index
    d = d[d[col_rbd].isin(validas)].copy()
    if covariables_escuela is not None:
        # Solo establecimientos con todas las covariables: el modelo no imputa.
        completas = covariables_escuela.dropna()
        antes = d[col_rbd].nunique()
        d = d[d[col_rbd].isin(completas.index)].copy()
        log.info("Covariables: %d de %d escuelas con datos completos.",
                 d[col_rbd].nunique(), antes)
    if exigir_2022:
        con_2022 = set(d.loc[d[col_anio] == ANIO_QUIEBRE, col_rbd])
        sin = d[col_rbd].nunique() - len(con_2022)
        if sin:
            log.info("%d escuelas excluidas por no tener puntaje en %d.",
                     sin, ANIO_QUIEBRE)
        d = d[d[col_rbd].isin(con_2022)].copy()
    if col_n is not None:
        d[col_n] = pd.to_numeric(d[col_n], errors="coerce")
        faltan = d[col_n].isna() | (d[col_n] <= 0)
        if faltan.any():
            # Sin dato de evaluados se imputa la mediana del establecimiento:
            # mantiene la observación sin inventar precisión.
            med = d.groupby(col_rbd)[col_n].transform("median")
            d.loc[faltan, col_n] = med[faltan]
            d[col_n] = d[col_n].fillna(d[col_n].median())

    rbds = np.sort(d[col_rbd].unique())
    idx = {r: i for i, r in enumerate(rbds)}

    grupo_escuela = (d.groupby(col_rbd)[col_grupo]
                     .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
                     .reindex(rbds))
    mapa_g = {g: i for i, g in enumerate(orden_grupo)}
    g_idx = grupo_escuela.map(mapa_g)
    sin_grupo = int(g_idx.isna().sum())
    if sin_grupo:
        log.warning("%d escuelas sin grupo reconocido; se asignan a la "
                    "categoría central.", sin_grupo)
    g_idx = g_idx.fillna(len(orden_grupo) // 2).astype(int).to_numpy()

    y = d[col_y].to_numpy(dtype="float64")
    media = float(y.mean())
    post = (d[col_anio] >= ANIO_QUIEBRE).to_numpy().astype("float64")
    t_post = np.where(post == 1, d[col_anio] - ANIO_QUIEBRE, 0.0)
    t_pre = np.where(post == 0, d[col_anio] - 2016.5, 0.0)

    log.info("Datos del modelo: %d escuelas, %d observaciones.",
             len(rbds), len(d))
    return {
        "escuela": d[col_rbd].map(idx).to_numpy(),
        "n": (d[col_n].to_numpy(dtype="float64") if col_n is not None else None),
        # Nivel previo de cada escuela, centrado y en unidades de 50 puntos.
        # Permite contrastar el gradiente socioeconómico de la pendiente a igual
        # nivel de partida (H3), dentro del propio modelo.
        "X_esc": (None if covariables_escuela is None else
                  _estandarizar(covariables_escuela.reindex(np.sort(d[col_rbd].unique())))),
        "nombres_cov": (None if covariables_escuela is None else list(covariables_escuela.columns)),
        # Desviación estándar y carácter binario de cada covariable, para
        # traducir los coeficientes a unidades interpretables.
        "de_cov": (None if covariables_escuela is None else
                   covariables_escuela.reindex(np.sort(d[col_rbd].unique())).astype(float).std().to_numpy()),
        "binaria_cov": (None if covariables_escuela is None else
                        [set(covariables_escuela[c].dropna().unique()) <= {0, 1, 0.0, 1.0}
                         for c in covariables_escuela.columns]),
        "pre_esc": (lambda pre: ((pre - pre.mean()) / ESCALA_SIMCE).to_numpy())(
            d[d[col_anio] < ANIO_QUIEBRE].groupby(col_rbd)[col_y].mean()
             .reindex(np.sort(d[col_rbd].unique())).fillna(
                 d[d[col_anio] < ANIO_QUIEBRE][col_y].mean())),
        "post": post, "t_post": t_post, "t_pre": t_pre,
        "anio": d[col_anio].to_numpy(),
        "grupo": g_idx,
        "y": (y - media) / ESCALA_SIMCE,
        "n_escuelas": len(rbds), "n_grupos": len(orden_grupo),
        "rbd": rbds, "nombres_grupo": list(orden_grupo),
        "media_y": media,
    }


def modelo_serie(escuela, post, t_post, t_pre, grupo, n_escuelas, n_grupos,
                 y=None, tendencia_pre: bool = False, n=None, pre_esc=None, X_esc=None):
    """Especificación NumPyro del modelo jerárquico de serie interrumpida.

    Parametrización mixta, elegida según cuánta información aporta cada
    escuela a cada parámetro:

    - **Intercepto centrado.** Con cuatro años PRE por escuela, el nivel
      previo está bien determinado por los datos y la forma centrada mezcla
      mejor.
    - **Caída y pendiente no centradas** (``media + desviación * z``). Dependen
      de solo cuatro años POST; con datos débiles la forma centrada produce la
      geometría de "embudo" que hace divergir a NUTS, y la no centrada la
      evita, como recomienda el tutorial de referencia.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 1.0))
    s_a = numpyro.sample("s_a", dist.HalfNormal(1.0))
    # Efectos de grupo sobre la caída y la recuperación (escala estandarizada).
    th_d = numpyro.sample("th_d", dist.Normal(0.0, 0.5).expand([n_grupos]))
    th_b = numpyro.sample("th_b", dist.Normal(0.0, 0.2).expand([n_grupos]))
    s_d = numpyro.sample("s_d", dist.HalfNormal(0.5))
    s_b = numpyro.sample("s_b", dist.HalfNormal(0.2))

    with numpyro.plate("escuelas", n_escuelas):
        # Intercepto CENTRADO: cada escuela tiene cuatro años PRE, de modo que
        # el nivel previo está bien informado por los datos. En ese régimen la
        # forma centrada mezcla mejor; la no centrada deja a mu_a mal
        # identificado (se probó: R-hat 1,05 y tamaño efectivo 69).
        a = numpyro.sample("a", dist.Normal(mu_a, s_a))
        # Caída y pendiente NO centradas: dependen de pocos años POST, y ahí
        # la forma no centrada evita la geometría de embudo.
        zd = numpyro.sample("zd", dist.Normal(0.0, 1.0))
        zb = numpyro.sample("zb", dist.Normal(0.0, 1.0))

    dlt = th_d[grupo] + s_d * zd
    b = th_b[grupo] + s_b * zb
    if X_esc is not None:
        # Covariables estandarizadas (media 0, DE 1): cada coeficiente es el
        # cambio, en unidades de 50 puntos, por una desviación estándar.
        p = X_esc.shape[1]
        gam_d = numpyro.sample("gam_d", dist.Normal(0.0, 0.5).expand([p]))
        gam_b = numpyro.sample("gam_b", dist.Normal(0.0, 0.2).expand([p]))
        dlt = dlt + X_esc @ gam_d
        b = b + X_esc @ gam_b
    if pre_esc is not None:
        # Con el nivel previo en la ecuación, th_b[g] es la pendiente del grupo g
        # a igual puntaje de partida: es el contraste de H3 sin el artefacto de
        # regresionar medias a posteriori ya contraídas hacia su grupo.
        g_d = numpyro.sample("g_d", dist.Normal(0.0, 0.5))
        g_b = numpyro.sample("g_b", dist.Normal(0.0, 0.2))
        dlt = dlt + g_d * pre_esc
        b = b + g_b * pre_esc
    numpyro.deterministic("caida", dlt * ESCALA_SIMCE)
    numpyro.deterministic("recuperacion", b * ESCALA_SIMCE)

    mu = a[escuela] + dlt[escuela] * post + b[escuela] * t_post
    if tendencia_pre:
        # Pendiente previa por grupo: contrasta el supuesto de PRE plano.
        th_p = numpyro.sample("th_p", dist.Normal(0.0, 0.2).expand([n_grupos]))
        numpyro.deterministic("tendencia_pre", th_p * ESCALA_SIMCE)
        mu = mu + th_p[grupo[escuela]] * t_pre

    if n is None:
        s_y = numpyro.sample("s_y", dist.HalfNormal(1.0))
        numpyro.sample("obs", dist.Normal(mu, s_y), obs=y)
    else:
        # Modelo de medición. Prior de s_w centrado cerca de 1: la desviación
        # individual de la escala SIMCE es 50 puntos, es decir, 1 en la escala
        # estandarizada. s_e recoge la variación del establecimiento entre
        # años que no es error de muestreo.
        s_w = numpyro.sample("s_w", dist.LogNormal(0.0, 0.3))
        s_e = numpyro.sample("s_e", dist.HalfNormal(0.5))
        sd = jnp.sqrt(s_w ** 2 / n + s_e ** 2)
        numpyro.sample("obs", dist.Normal(mu, sd), obs=y)


def ajustar(
    datos: dict,
    *,
    tendencia_pre: bool = False,
    controlar_pre: bool = False,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    semilla: int = 0,
    target_accept: float = 0.85,
):
    """Ajusta el modelo con NUTS y reporta los diagnósticos de convergencia.

    Con 2.866 escuelas y dos cadenas, el ajuste tarda unos 2-3 minutos en CPU.
    No se requiere GPU.

    Parameters
    ----------
    datos : dict
        Salida de :func:`preparar_datos`.
    tendencia_pre : bool, optional
    num_warmup, num_samples, num_chains : int, optional
        Dos cadenas es el mínimo para que el R-hat sea informativo.
    semilla : int, optional
    target_accept : float, optional

    Returns
    -------
    numpyro.infer.MCMC
        Objeto ajustado; ``mcmc.diagnosticos`` contiene divergencias, R-hat
        máximo y tamaño efectivo mínimo de la muestra.
    """
    import jax
    import numpyro
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(num_chains)
    kernel = NUTS(modelo_serie, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="sequential",
                progress_bar=False)

    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(semilla),
             datos["escuela"], datos["post"], datos["t_post"], datos["t_pre"],
             datos["grupo"], datos["n_escuelas"], datos["n_grupos"],
             y=datos["y"], tendencia_pre=tendencia_pre, n=datos.get("n"),
             pre_esc=datos.get("pre_esc") if controlar_pre else None,
             X_esc=datos.get("X_esc"),
             extra_fields=("diverging",))
    segundos = time.time() - t0

    por_cadena = mcmc.get_samples(group_by_chain=True)
    claves = [k for k in ("th_d", "th_b", "th_p", "g_d", "g_b", "gam_d", "gam_b", "s_d", "s_b", "s_y", "s_w", "s_e",
                          "mu_a", "s_a")
              if k in por_cadena]
    sm = summary({k: por_cadena[k] for k in claves})
    mcmc.diagnosticos = {
        "segundos": round(segundos, 1),
        "divergencias": int(mcmc.get_extra_fields()["diverging"].sum()),
        "r_hat_max": round(float(max(np.max(v["r_hat"]) for v in sm.values())), 3),
        "n_eff_min": int(min(np.min(v["n_eff"]) for v in sm.values())),
    }
    log.info("Ajuste: %s", mcmc.diagnosticos)
    if mcmc.diagnosticos["divergencias"] > 0:
        log.warning("Hay transiciones divergentes: los resultados pueden estar "
                    "sesgados. Sube target_accept (p. ej. 0.95).")
    if mcmc.diagnosticos["r_hat_max"] > 1.01:
        log.warning("R-hat > 1,01: las cadenas no convergieron del todo. "
                    "Aumenta num_warmup y num_samples.")
    return mcmc


def _intervalo(m: np.ndarray, prob: float = 0.94) -> tuple[np.ndarray, ...]:
    a = (1 - prob) / 2 * 100
    return (m.mean(0), np.percentile(m, a, 0), np.percentile(m, 100 - a, 0))


def resumen_grupos(mcmc, datos: dict, prob: float = 0.94) -> pd.DataFrame:
    """Efectos por grupo sobre la caída y la recuperación, en puntos SIMCE.

    Returns
    -------
    pandas.DataFrame
        Por grupo: media e intervalo de credibilidad de la caída inicial y de
        la recuperación anual (y de la tendencia previa, si se estimó).
    """
    s = mcmc.get_samples()
    out = pd.DataFrame({"grupo": datos["nombres_grupo"]})
    pares = [("th_d", "caida"), ("th_b", "recuperacion"), ("th_p", "tendencia_pre")]
    for clave, nombre in pares:
        if clave not in s:
            continue
        m, lo, hi = _intervalo(np.asarray(s[clave]) * ESCALA_SIMCE, prob)
        out[f"{nombre}"] = m.round(1)
        out[f"{nombre}_ic_inf"] = lo.round(1)
        out[f"{nombre}_ic_sup"] = hi.round(1)
    return out


def resumen_escuelas(mcmc, datos: dict, prob: float = 0.94) -> pd.DataFrame:
    """Caída y recuperación de cada escuela, con su intervalo de credibilidad.

    La columna ``recuperacion_creible`` indica si el intervalo excluye el
    cero, es decir, si la pendiente de esa escuela es distinguible del
    estancamiento con la información disponible.

    Returns
    -------
    pandas.DataFrame
        Una fila por RBD.
    """
    s = mcmc.get_samples()
    out = pd.DataFrame({"rbd": datos["rbd"]})
    for nombre in ("caida", "recuperacion"):
        m, lo, hi = _intervalo(np.asarray(s[nombre]), prob)
        out[f"{nombre}_bayes"] = m.round(2)
        out[f"{nombre}_ic_inf"] = lo.round(2)
        out[f"{nombre}_ic_sup"] = hi.round(2)
    out["recuperacion_creible"] = ((out["recuperacion_ic_inf"] > 0)
                                   | (out["recuperacion_ic_sup"] < 0))
    return out


# --------------------------------------------------------------------------- #
# Diagnósticos y medidas derivadas (revisión del 1-oct-2026)
# --------------------------------------------------------------------------- #

def chequeo_predictivo(mcmc, datos: dict, *, tendencia_pre: bool = False,
                       controlar_pre: bool = False,
                       semilla: int = 1, prob: float = 0.94) -> pd.DataFrame:
    """Chequeo predictivo posterior (comentario 19).

    Simula datos replicados desde el posterior y los compara con los
    observados. Un modelo bien especificado reproduce la media y la dispersión
    de cada año y cubre con su intervalo predictivo aproximadamente la
    proporción nominal de las observaciones.

    Returns
    -------
    pandas.DataFrame
        Por año: media y desviación observadas y replicadas (en puntos SIMCE)
        y cobertura del intervalo predictivo.
    """
    import jax
    from numpyro.infer import Predictive

    pred = Predictive(modelo_serie, posterior_samples=mcmc.get_samples())
    yrep = pred(jax.random.PRNGKey(semilla), datos["escuela"], datos["post"],
                datos["t_post"], datos["t_pre"], datos["grupo"],
                datos["n_escuelas"], datos["n_grupos"],
                tendencia_pre=tendencia_pre, n=datos.get("n"),
                pre_esc=datos.get("pre_esc") if controlar_pre else None,
                X_esc=datos.get("X_esc"))["obs"]
    yrep = np.asarray(yrep) * ESCALA_SIMCE + datos["media_y"]
    yobs = datos["y"] * ESCALA_SIMCE + datos["media_y"]
    a = (1 - prob) / 2 * 100
    lo, hi = np.percentile(yrep, a, 0), np.percentile(yrep, 100 - a, 0)
    cubre = (yobs >= lo) & (yobs <= hi)

    anio = datos["anio"]
    filas = []
    for t in np.unique(anio):
        m = anio == t
        filas.append({"anio": int(t),
                      "media_obs": round(float(yobs[m].mean()), 1),
                      "media_rep": round(float(yrep[:, m].mean()), 1),
                      "de_obs": round(float(yobs[m].std()), 1),
                      "de_rep": round(float(yrep[:, m].std(axis=1).mean()), 1),
                      "cobertura": round(float(cubre[m].mean()), 3)})
    out = pd.DataFrame(filas)
    log.info("Cobertura global del intervalo predictivo al %.0f %%: %.3f",
             prob * 100, float(cubre.mean()))
    return out


def correlacion_caida_pendiente(mcmc) -> dict:
    """Correlación entre caída y recuperación, descontado el ruido (comentario 11).

    Devuelve dos cantidades que no deben confundirse:

    - ``entre_escuelas``: correlación entre las medias a posteriori de la caída
      y la pendiente de cada establecimiento. Es la relación sustantiva.
    - ``dentro_escuela``: correlación media a posteriori entre ambos parámetros
      dentro de cada establecimiento. Mide cuánto comparten información (el
      puntaje de 2022); el modelo la reconoce en lugar de trasladarla a la
      relación entre escuelas.
    """
    s = mcmc.get_samples()
    d = np.asarray(s["caida"]); b = np.asarray(s["recuperacion"])
    entre = float(np.corrcoef(d.mean(0), b.mean(0))[0, 1])
    dc = d - d.mean(0); bc = b - b.mean(0)
    dentro = (dc * bc).mean(0) / (dc.std(0) * bc.std(0) + 1e-12)
    return {"entre_escuelas": round(entre, 3),
            "dentro_escuela_media": round(float(dentro.mean()), 3)}


def recuperacion_relativa_bayes(mcmc, datos: dict, *, prob_minima: float = 0.9,
                                horizonte: int = 3) -> pd.DataFrame:
    """Recuperación relativa con umbral de credibilidad (comentario 21).

    La razón entre lo recuperado y la caída es inestable cuando la caída es
    cercana a cero o positiva. Se calcula solo para los establecimientos cuya
    caída es creíblemente negativa: P(caída < 0) >= ``prob_minima``. El resto
    se clasifica como "sin caída detectable" y se reporta aparte.

    Lo recuperado se define como la pendiente por el horizonte (3 años, de
    2022 a 2025), no como la diferencia entre 2025 y 2022: la pendiente usa los
    cuatro años posteriores y es menos ruidosa que la diferencia entre dos
    puntos.
    """
    s = mcmc.get_samples()
    d = np.asarray(s["caida"]); b = np.asarray(s["recuperacion"])
    p_caida = (d < 0).mean(0)
    import warnings
    rho = np.where(d < 0, horizonte * b / np.maximum(-d, 1e-9), np.nan)
    warnings.filterwarnings("ignore", message="All-NaN slice")
    out = pd.DataFrame({"rbd": datos["rbd"], "prob_caida": p_caida.round(3)})
    elegible = p_caida >= prob_minima
    out["clase"] = np.where(elegible, "caída creíble", "sin caída detectable")
    med = np.nanmedian(rho, axis=0)
    lo = np.nanpercentile(rho, 3, axis=0); hi = np.nanpercentile(rho, 97, axis=0)
    out["rho_mediana"] = np.where(elegible, med, np.nan).round(3)
    out["rho_ic_inf"] = np.where(elegible, lo, np.nan).round(3)
    out["rho_ic_sup"] = np.where(elegible, hi, np.nan).round(3)
    log.info("Caída creíble en %d de %d establecimientos (P >= %.2f).",
             int(elegible.sum()), len(out), prob_minima)
    return out


def _estandarizar(X: pd.DataFrame) -> np.ndarray:
    """Centra y escala cada covariable (las indicadoras 0/1 también, para que los
    priors tengan la misma escala en todos los coeficientes)."""
    Z = X.astype(float)
    sd = Z.std().replace(0, 1.0)
    return ((Z - Z.mean()) / sd).to_numpy()


def resumen_covariables(mcmc, datos: dict, prob: float = 0.94) -> pd.DataFrame:
    """Efecto de cada covariable sobre la caída y la recuperación.

    Las continuas se expresan por una desviación estándar; las indicadoras
    (0/1), como el contraste entre 0 y 1, que es su unidad natural. Un intervalo
    que excluye el cero indica una asociación creíble, condicional a las demás
    variables del modelo.

    Con muchas covariables y dos resultados, algunos intervalos excluirán el
    cero por azar: con 16 intervalos al 94 %, cabe esperar uno. Las asociaciones
    al borde del cero deben leerse con esa cautela.
    """
    s = mcmc.get_samples()
    if "gam_d" not in s:
        raise ValueError("El modelo no se ajustó con covariables.")
    a = (1 - prob) / 2 * 100
    filas = []
    binarias = datos.get("binaria_cov") or [False] * len(datos["nombres_cov"])
    de = datos.get("de_cov")
    for j, nombre in enumerate(datos["nombres_cov"]):
        es_bin = bool(binarias[j])
        fila = {"covariable": nombre, "unidad": "0 → 1" if es_bin else "+1 DE"}
        for clave, etiqueta in (("gam_d", "caida"), ("gam_b", "recuperacion")):
            v = np.asarray(s[clave])[:, j] * ESCALA_SIMCE
            if es_bin and de is not None and de[j] > 0:
                v = v / de[j]         # el modelo usa la indicadora estandarizada
            fila[etiqueta] = round(float(np.median(v)), 2)
            fila[f"{etiqueta}_ic"] = [round(float(np.percentile(v, a)), 2),
                                      round(float(np.percentile(v, 100 - a)), 2)]
            fila[f"{etiqueta}_creible"] = bool(np.percentile(v, a) > 0 or np.percentile(v, 100 - a) < 0)
        filas.append(fila)
    return pd.DataFrame(filas)


def gradiente_gse(mcmc, alto: int = 4, bajo: int = 0, prob: float = 0.94) -> dict:
    """Diferencia de pendiente entre GSE Alto y Bajo, con intervalo y probabilidad."""
    tb = np.asarray(mcmc.get_samples()["th_b"]) * ESCALA_SIMCE
    v = tb[:, alto] - tb[:, bajo]
    a = (1 - prob) / 2 * 100
    return {"alto_menos_bajo": round(float(np.median(v)), 2),
            "ic": [round(float(np.percentile(v, a)), 2), round(float(np.percentile(v, 100 - a)), 2)],
            "prob_positivo": round(float((v > 0).mean()), 3)}


# --------------------------------------------------------------------------- #
# H4: contribución de bloques de covariables por subgrupo
# --------------------------------------------------------------------------- #

def agregar_interacciones(X: pd.DataFrame, moderadora: str, columnas: list[str], *,
                          min_casos: int = 10) -> pd.DataFrame:
    """Agrega interacciones ``columna × moderadora`` a la matriz de covariables.

    Una interacción con menos de ``min_casos`` valores distintos de cero (por
    ejemplo, particulares pagados rurales) no tiene información para estimarse
    y solo agregaría un coeficiente determinado por el prior: se omite.
    """
    out = X.copy()
    omitidas = []
    for c in columnas:
        inter = X[c] * X[moderadora]
        if (inter.fillna(0) != 0).sum() < min_casos:
            omitidas.append(c)
            continue
        out[f"{c}__x_{moderadora}"] = inter
    if omitidas:
        log.info("Interacciones omitidas por falta de casos: %s", omitidas)
    return out


def contribucion_bloques(mcmc, datos: dict, bloques: dict[str, list[str]],
                         subgrupo: np.ndarray, *, parametro: str = "gam_b",
                         prob: float = 0.94, corregir_ruido: bool = True) -> pd.DataFrame:
    """Contribución de cada bloque de covariables a la variación entre escuelas.

    Para cada muestra del posterior y cada bloque se calcula la varianza, entre
    las escuelas del subgrupo, de la parte del predictor lineal que aporta el
    bloque (sus efectos principales más sus interacciones). Es el análogo, en un
    modelo lineal, de la importancia SHAP: cuánta variación de la pendiente (o
    de la caída) entre establecimientos se debe a ese conjunto de variables.

    Parameters
    ----------
    bloques : dict
        Nombre del bloque -> covariables que lo componen. Las interacciones
        ``<covariable>__x_<moderadora>`` se asignan automáticamente al bloque de
        su covariable.
    subgrupo : numpy.ndarray de bool
        Escuelas, en el orden de ``datos["rbd"]``, sobre las que se calcula.
    parametro : str
        ``"gam_b"`` para la recuperación, ``"gam_d"`` para la caída.
    corregir_ruido : bool
        Resta el piso de ruido de cada bloque. Sin la corrección, el bloque con
        más coeficientes inciertos parece contribuir más aunque no lo haga (se
        verificó con datos simulados bajo la hipótesis nula).

    Returns
    -------
    pandas.DataFrame
        Por bloque: desviación estándar de su contribución (puntos o puntos por
        año), con intervalo, y las muestras en ``attrs["muestras"]``.
    """
    s = np.asarray(mcmc.get_samples()[parametro]) * ESCALA_SIMCE
    X = datos["X_esc"][subgrupo]
    nombres = datos["nombres_cov"]
    a = (1 - prob) / 2 * 100
    muestras, filas = {}, []
    for bloque, cols in bloques.items():
        idx = [j for j, n in enumerate(nombres)
               if n in cols or n.split("__x_")[0] in cols]
        if not idx:
            continue
        comp = X[:, idx] @ s[:, idx].T            # escuelas × muestras
        var = comp.var(axis=0)                    # una varianza por muestra
        if corregir_ruido:
            # La incertidumbre de los coeficientes infla la varianza de cada
            # muestra: E[Var(X g)] = Var(X g_medio) + E[Var(X (g - g_medio))].
            # El segundo término es un piso de ruido propio del bloque, mayor
            # cuantos más coeficientes inciertos tiene. Se resta, para que dos
            # bloques de distinto tamaño se comparen en igualdad de condiciones.
            centro = X[:, idx] @ s[:, idx].mean(axis=0)
            piso = (comp - centro[:, None]).var(axis=0).mean()
            var = np.maximum(var - piso, 0.0)
        de = np.sqrt(var)
        muestras[bloque] = de
        filas.append({"bloque": bloque, "covariables": len(idx),
                      "de_contribucion": round(float(np.median(de)), 3),
                      "ic": [round(float(np.percentile(de, a)), 3),
                             round(float(np.percentile(de, 100 - a)), 3)]})
    out = pd.DataFrame(filas)
    out.attrs["muestras"] = muestras
    return out


def decision_h4(rural: pd.DataFrame, urbano: pd.DataFrame, *, umbral: float = 0.90) -> dict:
    """Aplica la regla de decisión de H4, fijada antes de ver los resultados.

    H4: en los establecimientos rurales las variables geográficas contribuyen
    más que la dependencia; en los urbanos, al revés.

    - Se sostiene si P(geo > dep | rural) >= umbral y P(dep > geo | urbano) >= umbral.
    - Se refuta si ambas probabilidades son <= 1 - umbral.
    - En cualquier otro caso, es no concluyente.
    """
    mr, mu = rural.attrs["muestras"], urbano.attrs["muestras"]
    p_rural = float((mr["geograficas"] > mr["dependencia"]).mean())
    p_urbano = float((mu["dependencia"] > mu["geograficas"]).mean())
    if p_rural >= umbral and p_urbano >= umbral:
        veredicto = "se sostiene"
    elif p_rural <= 1 - umbral and p_urbano <= 1 - umbral:
        veredicto = "se refuta"
    else:
        veredicto = "no concluyente"
    return {"P(geo > dep | rural)": round(p_rural, 3),
            "P(dep > geo | urbano)": round(p_urbano, 3),
            "umbral": umbral, "veredicto": veredicto}


def simular_h4(n_escuelas: int = 1500, *, h4_verdadera: bool = True, prop_rural: float = 0.054,
               semilla: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel simulado para validar el contraste de H4.

    Con ``h4_verdadera=True``, la geografía mueve la pendiente en los rurales y
    la dependencia en los urbanos. Con ``False`` (hipótesis nula), ambos bloques
    contribuyen parecido en los dos grupos. La proporción de rurales y la
    estructura de bloques imitan las de la tesis: tres variables geográficas,
    tres de dependencia, y sin particulares pagados rurales.

    Returns
    -------
    tuple
        ``(panel_largo, covariables)`` listos para :func:`preparar_datos`.
    """
    rng = np.random.default_rng(semilla)
    N = n_escuelas; n_rur = int(round(prop_rural * N))
    rural = np.zeros(N); rural[:n_rur] = 1
    geo = rng.normal(size=(N, 3))
    dep = (rng.random((N, 3)) < [0.35, 0.10, 0.15]).astype(float); dep[:n_rur, 2] = 0
    if h4_verdadera:
        efecto = (np.where(rural == 1, 2.0 * geo[:, 0] + 1.5 * geo[:, 1], 0.2 * geo[:, 0])
                  + np.where(rural == 1, 0.2 * dep[:, 0], 3.0 * dep[:, 0] - 2.5 * dep[:, 1]))
    else:
        efecto = (0.4 * geo[:, 0] + 0.3 * geo[:, 1] + 0.3 * geo[:, 2]
                  + 1.0 * dep[:, 0] - 0.8 * dep[:, 1] + 0.6 * dep[:, 2])
    b = 3.5 + efecto + rng.normal(0, 1.5, N)
    a = 260 + rng.normal(0, 35, N); d = -10 + rng.normal(0, 12, N)
    filas = []
    for i in range(N):
        n = rng.integers(15, 120)
        for t in (2015, 2016, 2017, 2018, 2022, 2023, 2024, 2025):
            mu = a[i] + (d[i] + b[i] * (t - 2022) if t >= 2022 else 0)
            filas.append((i, t, mu + rng.normal(0, 50 / np.sqrt(n) + 4), n))
    L = pd.DataFrame(filas, columns=["rbd", "anio", "y", "n"]); L["cod_grupo_cat"] = "Medio"
    X = pd.DataFrame({"geo1": geo[:, 0], "geo2": geo[:, 1], "geo3": geo[:, 2],
                      "dep1": dep[:, 0], "dep2": dep[:, 1], "dep3": dep[:, 2], "rural": rural},
                     index=range(N))
    X = agregar_interacciones(X, "rural", ["geo1", "geo2", "geo3", "dep1", "dep2", "dep3"])
    return L, X


def diagnosticos_escuela(mcmc) -> dict:
    """R-hat máximo y tamaño efectivo mínimo de la caída y la recuperación de
    cada escuela (los diagnósticos de :func:`ajustar` cubren solo los
    parámetros de grupo y de varianza)."""
    from numpyro.diagnostics import summary
    pc = mcmc.get_samples(group_by_chain=True)
    sm = summary({k: pc[k] for k in ("caida", "recuperacion") if k in pc})
    return {k: {"r_hat_max": round(float(np.max(v["r_hat"])), 3),
                "n_eff_min": int(np.min(v["n_eff"]))} for k, v in sm.items()}


def gse_modal_previo(largo: pd.DataFrame, *, col_rbd: str = "rbd", col_anio: str = "año",
                     col_gse: str = "cod_grupo_cat",
                     anios: tuple[int, ...] = (2015, 2016, 2017, 2018)) -> pd.Series:
    """GSE modal de cada establecimiento en el período previo a la pandemia.

    Añadida el 24-oct-2026. Los cuadernos asignaban a cada escuela el GSE del
    año más reciente, que es posterior al tratamiento: una escuela que cambió
    de grupo después de 2022 quedaba clasificada según su situación posterior.
    En el panel, 719 de 2.861 escuelas del modelo (25 %) tienen un GSE modal
    2015-2018 distinto del más reciente. Úsese para las pruebas de
    sensibilidad de H3 y H3b.

    ``largo`` es el panel largo por establecimiento y año, con la columna
    canónica del GSE (``estandarizar_canonico(largo, variables=["cod_grupo"])``).
    """
    d = largo[pd.to_numeric(largo[col_anio], errors="coerce").isin(anios)].dropna(subset=[col_gse])
    d = d.sort_values(col_anio)

    def moda(g: pd.DataFrame):
        # Empates (por ejemplo, dos años en cada grupo): gana el grupo del año
        # previo más reciente, no el primero en orden alfabético.
        vc = g[col_gse].value_counts()
        top = set(vc.index[vc == vc.max()])
        return g.loc[g[col_gse].isin(top), col_gse].iloc[-1]

    return d.groupby(col_rbd)[[col_anio, col_gse]].apply(moda).rename("gse_previo")
