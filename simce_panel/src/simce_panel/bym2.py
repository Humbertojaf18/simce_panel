"""
bym2.py
=======

Modelo BYM2 por comuna (revisión del 1-oct-2026, comentarios 10, 11 y 39).

Por qué
-------
LISA contrasta establecimiento por establecimiento, y con pendientes tan
inciertas no puede detectar conglomerados. El BYM2 traslada la pregunta al
nivel donde sí hay información: estima un efecto por comuna y descompone su
varianza en una parte estructurada espacialmente (comunas vecinas se parecen)
y otra no estructurada. El parámetro rho estima directamente la fracción de la
varianza comunal con estructura espacial (Riebler et al., 2016).

    y_i = X_i beta + phi_c(i) + e_i
    phi = sigma_phi * ( sqrt(rho) * u / sqrt(s) + sqrt(1 - rho) * v )

donde u es un efecto ICAR (autorregresivo intrínseco) sobre el grafo de
comunas, v un efecto independiente y s el factor de escala que deja la parte
espacial con varianza media 1, comparable con v. Como el grafo de vecinos más
cercanos tiene varios componentes conexos, cada componente lleva su propio
factor s y su propia restricción de suma cero, y las comunas sin vecinos solo
tienen parte independiente (Freni-Sterrantino et al., 2018).

Grafo de comunas
----------------
No se dispone de los polígonos comunales, de modo que la vecindad se define por
los k centroides más cercanos (centroide de los establecimientos de cada
comuna), simetrizada. Las islas oceánicas quedan unidas a sus vecinas más
próximas del continente; con k pequeño su influencia es mínima.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def grafo_comunas(df: pd.DataFrame, *, k: int = 5, col_comuna: str = "comuna_dir",
                  col_lat: str = "latitud", col_lon: str = "longitud"):
    """Construye el grafo de vecindad entre comunas y sus factores de escala.

    Corrección del 24-oct-2026. La versión anterior calculaba el factor de
    escala con la inversa de Q más una pequeña constante en la diagonal: esa
    constante convierte la dirección sin información del ICAR (la media de cada
    componente) en una varianza enorme, y el factor salía unas 11.500 veces más
    grande de lo debido (8.962 en lugar de 0,78). Con ese factor la parte
    espacial del efecto comunal quedaba anulada y rho no podía estimarse.

    Ahora, siguiendo a Riebler et al. (2016) y Freni-Sterrantino et al. (2018):

    - se identifican los componentes conexos del grafo;
    - cada componente con más de una comuna recibe su propio factor de escala,
      la media geométrica de las varianzas marginales de la inversa generalizada
      de su matriz Q, y su propia restricción de suma cero;
    - las comunas aisladas (sin vecinos) no tienen parte espacial: su efecto es
      solo la componente independiente.

    Returns
    -------
    dict
        ``comunas``, ``nodo1``/``nodo2``, ``n_comunas``, ``componente`` (por
        comuna), ``escala_nodo`` (factor de su componente), ``aislada`` y
        ``escala`` (media geométrica de los factores, solo para reportar).
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors

    c = df.dropna(subset=[col_comuna, col_lat, col_lon]).groupby(col_comuna)[[col_lat, col_lon]].mean()
    R = 6371.0
    lat0 = np.radians(c[col_lat].mean())
    xy = np.column_stack([R * np.radians(c[col_lon]) * np.cos(lat0), R * np.radians(c[col_lat])])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, ind = nn.kneighbors(xy)
    aristas = {tuple(sorted((i, j))) for i in range(len(c)) for j in ind[i, 1:]}
    n1 = np.array([a for a, _ in aristas]); n2 = np.array([b for _, b in aristas])

    n = len(c)
    W = np.zeros((n, n)); W[n1, n2] = 1; W[n2, n1] = 1
    Q = np.diag(W.sum(1)) - W
    n_comp, comp = connected_components(W, directed=False)
    escala_nodo = np.ones(n)
    aislada = np.zeros(n, dtype=bool)
    for kk in range(n_comp):
        idx = np.where(comp == kk)[0]
        if len(idx) == 1:
            aislada[idx] = True
            continue
        # Inversa generalizada exacta del laplaciano de un grafo conexo:
        # (Q + 11'/m)^-1 - 11'/m. pinv con la tolerancia por defecto puede
        # invertir el autovalor nulo (del orden de 1e-14) y devolver basura.
        m = len(idx)
        J = np.full((m, m), 1.0 / m)
        Qp = np.linalg.inv(Q[np.ix_(idx, idx)] + J) - J
        escala_nodo[idx] = float(np.exp(np.mean(np.log(np.diag(Qp)))))
    escala = float(np.exp(np.mean(np.log(escala_nodo[~aislada])))) if (~aislada).any() else 1.0
    log.info("Grafo: %d comunas, %d aristas, %d componentes conexos (%d aisladas); "
             "factor de escala medio %.3f.", n, len(aristas), n_comp, int(aislada.sum()), escala)
    return {"comunas": c.index.to_numpy(), "nodo1": n1, "nodo2": n2, "n_comunas": n,
            "componente": comp, "n_componentes": int(n_comp), "escala_nodo": escala_nodo,
            "aislada": aislada, "escala": escala}


def _modelo(y, X, comuna, nodo1, nodo2, n_comunas, escala_nodo, componente, aislada):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    beta0 = numpyro.sample("beta0", dist.Normal(0, 1))
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1]]))
    sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(1.0))
    rho = numpyro.sample("rho", dist.Beta(0.5, 0.5))
    # ICAR propio (sin la normal estándar que lo volvía otro prior).
    u = numpyro.sample("u", dist.ImproperUniform(dist.constraints.real, (), event_shape=(n_comunas,)))
    numpyro.factor("icar", -0.5 * jnp.sum((u[nodo1] - u[nodo2]) ** 2))
    # Una restricción de suma cero por componente conexo.
    n_comp = int(componente.max()) + 1
    sumas = jnp.zeros(n_comp).at[componente].add(jnp.where(aislada, 0.0, u))
    tam = np.bincount(componente, minlength=n_comp)
    numpyro.factor("suma_cero", dist.Normal(0, 0.001 * jnp.asarray(tam)).log_prob(sumas).sum())
    v = numpyro.sample("v", dist.Normal(0, 1).expand([n_comunas]))
    # Las comunas aisladas no tienen parte espacial: usan una normal propia.
    w_ais = numpyro.sample("w_ais", dist.Normal(0, 1).expand([n_comunas]))
    espacial = jnp.where(aislada, w_ais, u / jnp.sqrt(escala_nodo))
    phi = sigma_phi * (jnp.sqrt(rho) * espacial + jnp.sqrt(1 - rho) * v)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(beta0 + X @ beta + phi[comuna], sigma), obs=y)


def ajustar_bym2(df: pd.DataFrame, columna: str, covariables: list[str], grafo: dict, *,
                 col_comuna: str = "comuna_dir", num_warmup: int = 2000,
                 num_samples: int = 3000, num_chains: int = 4, semilla: int = 0) -> dict:
    """Ajusta el BYM2 y resume rho y la fracción de varianza comunal.

    ``columna`` se estandariza; las covariables categóricas se codifican como
    indicadoras. Devuelve rho (fracción espacial de la varianza comunal), la
    fracción de la varianza total atribuible a la comuna y los diagnósticos.

    Por defecto usa cuatro cadenas de 2.000 + 3.000 iteraciones. Con la escala
    corregida, rho se acerca al borde superior y mezcla lento: con 1.500 + 1.500
    su R-hat quedó en 1,011-1,012 y su tamaño efectivo en 240-390 en Windows
    (versión 1.1.0). Con 3.000 muestras por cadena el R-hat queda bajo 1,01 y
    el tamaño efectivo sobre 400. Tarda unos 6 minutos por variable.
    """
    import jax
    import numpyro
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    d = df.dropna(subset=[columna, col_comuna] + covariables).copy()
    d = d[d[col_comuna].isin(grafo["comunas"])]
    y = d[columna].to_numpy(float); y = (y - y.mean()) / y.std()
    X = pd.get_dummies(d[covariables], drop_first=True, dtype=float)
    X = ((X - X.mean()) / X.std().replace(0, 1)).to_numpy(float)
    pos = {c: i for i, c in enumerate(grafo["comunas"])}
    comuna = d[col_comuna].map(pos).to_numpy()

    numpyro.set_host_device_count(num_chains)
    mcmc = MCMC(NUTS(_modelo, target_accept_prob=0.9), num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(semilla), y, X, comuna, grafo["nodo1"], grafo["nodo2"],
             grafo["n_comunas"], grafo["escala_nodo"], grafo["componente"], grafo["aislada"],
             extra_fields=("diverging",))
    s = mcmc.get_samples()
    pc = mcmc.get_samples(group_by_chain=True)
    sm = summary({k: pc[k] for k in ("rho", "sigma_phi", "sigma", "beta0")})
    frac = s["sigma_phi"] ** 2 / (s["sigma_phi"] ** 2 + s["sigma"] ** 2)
    q = lambda a: [round(float(np.percentile(a, p)), 3) for p in (3, 50, 97)]
    return {"variable": columna, "n": len(d), "comunas": int(len(np.unique(comuna))),
            "rho_[3,50,97]": q(s["rho"]),
            "fraccion_varianza_comunal_[3,50,97]": q(frac),
            "divergencias": int(mcmc.get_extra_fields()["diverging"].sum()),
            "r_hat_max": round(float(max(np.max(v["r_hat"]) for v in sm.values())), 3),
            "r_hat_rho": round(float(sm["rho"]["r_hat"]), 3),
            "ess_rho": int(sm["rho"]["n_eff"]),
            "segundos": round(time.time() - t0, 1)}
