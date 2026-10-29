"""
panel_ancho.py
==============

Transformación del panel **largo** (una fila por establecimiento-año) al panel
**ancho**: una única fila por establecimiento (RBD), con las variables anuales
desplegadas en columnas ``<variable>_<año>``.

Criterio de separación de variables
-----------------------------------
No todas las columnas deben pivotarse. Se distinguen tres grupos:

- **Identidad** (``rbd``): la llave; una sola columna.
- **Atributos de ficha** (nombre, región, comuna, dependencia, GSE...): cambian
  poco, pero *pueden* cambiar (un colegio se traspasa a un SLEP, Ñuble se
  separa de Biobío en 2018). Se resuelven con una política explícita
  (:data:`POLITICAS_ATRIBUTO`): por defecto se toma el valor del año más
  reciente disponible (``ultimo``) y, además, se marca si el atributo varió a
  lo largo de la serie (``<atributo>_cambio``), para no perder esa información.
- **Medidas anuales** (puntajes, número de alumnos, diferencias, marcas...):
  se pivotan a ``<variable>_<año>``.

Funciones principales
---------------------
- :func:`clasificar_columnas` : separa identidad / atributos / medidas.
- :func:`resolver_atributos`  : una fila por RBD con los atributos de ficha.
- :func:`pivotar_medidas`     : despliega las medidas anuales en columnas.
- :func:`construir_panel_ancho` : orquesta todo y devuelve la tabla final.
"""

from __future__ import annotations

import logging

import pandas as pd

from .columnas import normalizar_texto

log = logging.getLogger(__name__)

#: Columna llave del establecimiento.
COL_RBD: str = "rbd"

#: Prefijos/nombres que se consideran atributos de ficha (no se pivotan).
#: Se comparan por nombre normalizado; ``*`` indica coincidencia por prefijo.
ATRIBUTOS_FICHA: tuple[str, ...] = (
    "dvrbd", "nom_rbd", "grado",
    "cod_reg_rbd", "nom_reg_rbd", "cod_pro_rbd", "nom_pro_rbd",
    "cod_deprov_rbd", "nom_deprov_rbd", "cod_com_rbd", "nom_com_rbd",
    "cod_depe1", "cod_depe2", "cod_grupo", "cod_rural_rbd",
    "cod_depe1_desc", "cod_depe2_desc", "cod_grupo_desc", "cod_rural_rbd_desc",
    "cod_depe1_cat", "cod_depe2_cat", "cod_grupo_cat", "cod_rural_rbd_cat",
    "latitud", "longitud",
    # Territoriales tomadas del Directorio (más fiables que las del SIMCE).
    "region_dir", "comuna_dir", "dependencia_dir", "ruralidad_dir",
)

#: Columnas técnicas que no aportan al panel ancho y se descartan.
COLUMNAS_DESCARTABLES: tuple[str, ...] = ("fecha_bbdd", "codigo_bbdd", "agno")

#: Política por defecto para resolver un atributo de ficha a un único valor.
#: ``ultimo``: valor del año más reciente no nulo. ``moda``: el más frecuente.
POLITICAS_ATRIBUTO: dict[str, str] = {"_default": "ultimo"}


def _es_atributo(col: str) -> bool:
    n = normalizar_texto(col)
    return any(n == normalizar_texto(a) for a in ATRIBUTOS_FICHA)


def _es_descartable(col: str) -> bool:
    n = normalizar_texto(col)
    return any(n == normalizar_texto(a) for a in COLUMNAS_DESCARTABLES)


def clasificar_columnas(
    df: pd.DataFrame,
    *,
    col_rbd: str = COL_RBD,
    col_anio: str = "año",
) -> dict[str, list[str]]:
    """Separa las columnas del panel largo en identidad, atributos y medidas.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo.
    col_rbd, col_anio : str, optional

    Returns
    -------
    dict[str, list[str]]
        Claves ``identidad``, ``atributos``, ``medidas`` y ``descartadas``.
    """
    identidad, atributos, medidas, descartadas = [], [], [], []
    for c in df.columns:
        if c in (col_rbd, col_anio):
            identidad.append(c)
        elif _es_descartable(c):
            descartadas.append(c)
        elif _es_atributo(c):
            atributos.append(c)
        else:
            medidas.append(c)
    return {"identidad": identidad, "atributos": atributos,
            "medidas": medidas, "descartadas": descartadas}


def resolver_atributos(
    df: pd.DataFrame,
    atributos: list[str],
    *,
    col_rbd: str = COL_RBD,
    col_anio: str = "año",
    politica: str = "ultimo",
    marcar_cambios: bool = True,
    col_para_cambio: dict[str, str] | None = None,
    tolerancias: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reduce los atributos de ficha a una única fila por RBD.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo.
    atributos : list[str]
        Columnas de ficha a resolver.
    col_rbd, col_anio : str, optional
    politica : str, optional
        ``"ultimo"`` (valor del año más reciente no nulo) o ``"moda"``.
    marcar_cambios : bool, optional
        Si ``True``, añade ``<atributo>_cambio`` (booleano) indicando si el
        atributo tomó más de un valor distinto a lo largo de la serie. Es
        relevante para casos como el traspaso de sostenedor a un SLEP o el
        cambio de región por la creación de Ñuble.
    col_para_cambio : dict[str, str] | None, optional
        Columna alternativa sobre la que evaluar el cambio de un atributo.
        IMPRESCINDIBLE cuando la columna cruda mezcla formatos entre años (en
        esta serie, las categóricas vienen como texto hasta 2017 y como código
        desde 2018): comparar el valor crudo marcaría como "cambio" un simple
        cambio de codificación. Se debe apuntar a la columna canónica
        (``<atributo>_cat``).

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD.
    """
    if not atributos:
        return pd.DataFrame(index=pd.Index([], name=col_rbd))

    orden = df.sort_values(col_anio)
    g = orden.groupby(col_rbd)
    partes = {}
    for a in atributos:
        if politica == "moda":
            partes[a] = g[a].agg(
                lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else pd.NA)
        else:  # ultimo no nulo
            partes[a] = g[a].agg(
                lambda s: s.dropna().iloc[-1] if not s.dropna().empty else pd.NA)
    out = pd.DataFrame(partes)

    if marcar_cambios:
        ref = col_para_cambio or {}
        tol = tolerancias or {}
        for a in atributos:
            col = ref.get(a, a)
            if col not in df.columns:
                col = a
            if a in tol:
                # Atributo continuo: cambia solo si supera la tolerancia.
                limite = tol[a]
                serie = pd.to_numeric(orden[col], errors="coerce")
                tmp = pd.DataFrame({col_rbd: orden[col_rbd], "_v": serie})
                rango = (tmp.dropna(subset=["_v"])
                         .groupby(col_rbd)["_v"]
                         .agg(lambda s: s.max() - s.min()))
                out[f"{a}_cambio"] = (rango > limite).reindex(out.index).fillna(False)
            else:
                out[f"{a}_cambio"] = (orden.groupby(col_rbd)[col]
                                      .agg(lambda s: s.dropna().nunique() > 1))
    return out


def pivotar_medidas(
    df: pd.DataFrame,
    medidas: list[str],
    *,
    col_rbd: str = COL_RBD,
    col_anio: str = "año",
    anios: list[int] | None = None,
) -> pd.DataFrame:
    """Despliega las medidas anuales en columnas ``<variable>_<año>``.

    Si un establecimiento tuviera más de una fila en el mismo año (no debería
    ocurrir a nivel RBD), se conserva la primera y se registra una advertencia.

    Parameters
    ----------
    df : pandas.DataFrame
    medidas : list[str]
    col_rbd, col_anio : str, optional
    anios : list[int] | None, optional
        Años a incluir y su orden. Si es ``None``, se usan los presentes.

    Returns
    -------
    pandas.DataFrame
        Indexado por RBD, con columnas ordenadas por variable y año.
    """
    if not medidas:
        return pd.DataFrame(index=pd.Index([], name=col_rbd))

    d = df.copy()
    d[col_anio] = pd.to_numeric(d[col_anio], errors="coerce").astype("Int64")
    dup = d.duplicated(subset=[col_rbd, col_anio], keep=False)
    if dup.any():
        log.warning("%d filas duplicadas por (rbd, año); se conserva la primera.",
                    int(dup.sum()))
        d = d.drop_duplicates(subset=[col_rbd, col_anio], keep="first")

    ancho = d.pivot(index=col_rbd, columns=col_anio, values=medidas)
    # MultiIndex (variable, año) -> 'variable_año'
    anios_pres = anios or sorted(
        {a for _, a in ancho.columns if pd.notna(a)})
    columnas_ordenadas = []
    for var in medidas:
        for a in anios_pres:
            if (var, a) in ancho.columns:
                columnas_ordenadas.append((var, a))
    ancho = ancho[columnas_ordenadas]
    ancho.columns = [f"{var}_{int(a)}" for var, a in ancho.columns]
    return ancho


def construir_panel_ancho(
    df_largo: pd.DataFrame,
    *,
    col_rbd: str = COL_RBD,
    col_anio: str = "año",
    politica_atributos: str = "ultimo",
    marcar_cambios: bool = True,
    tolerancia_metros: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye el panel ancho: una fila por establecimiento.

    Parameters
    ----------
    df_largo : pandas.DataFrame
        Panel largo (salida de la unificación + estandarización).
    col_rbd, col_anio : str, optional
    politica_atributos : str, optional
        ``"ultimo"`` o ``"moda"`` (ver :func:`resolver_atributos`).
    marcar_cambios : bool, optional

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(panel_ancho, mapa_columnas)``. El mapa documenta, para cada columna
        final, la variable de origen, el año y el rol (identidad / atributo /
        medida), y alimenta el glosario consolidado.
    """
    clases = clasificar_columnas(df_largo, col_rbd=col_rbd, col_anio=col_anio)
    log.info("Columnas -> atributos: %d | medidas: %d | descartadas: %d",
             len(clases["atributos"]), len(clases["medidas"]),
             len(clases["descartadas"]))

    # El cambio de un atributo se evalúa sobre su columna canónica '<attr>_cat'
    # cuando existe, para no confundir un cambio de codificación (texto ->
    # código en 2018) con un cambio real de categoría.
    ref_cambio = {a: f"{a}_cat" for a in clases["atributos"]
                  if f"{a}_cat" in df_largo.columns}
    # Las coordenadas se comparan con tolerancia: el Directorio las
    # re-geocodifica cada año con precisión variable (un mismo colegio puede
    # aparecer como -18.4872 y -18.487245), lo que sin tolerancia marcaría
    # como "traslado" a casi todos los establecimientos.
    grados = tolerancia_metros / 111_000.0
    tol = {c: grados for c in ("latitud", "longitud")
           if c in clases["atributos"]}

    atrib = resolver_atributos(
        df_largo, clases["atributos"], col_rbd=col_rbd, col_anio=col_anio,
        politica=politica_atributos, marcar_cambios=marcar_cambios,
        col_para_cambio=ref_cambio, tolerancias=tol)
    med = pivotar_medidas(df_largo, clases["medidas"],
                          col_rbd=col_rbd, col_anio=col_anio)

    ancho = atrib.join(med, how="outer") if not atrib.empty else med
    ancho = ancho.reset_index()

    # Cobertura: en cuántos años aparece cada establecimiento.
    presencia = (df_largo.groupby(col_rbd)[col_anio]
                 .nunique().rename("n_anios_presente"))
    ancho = ancho.merge(presencia, on=col_rbd, how="left")

    # Mapa de columnas para el glosario consolidado.
    filas = [{"columna": col_rbd, "variable_origen": col_rbd,
              "año": None, "rol": "identidad"}]
    for a in clases["atributos"]:
        filas.append({"columna": a, "variable_origen": a,
                      "año": None, "rol": "atributo de ficha"})
        if marcar_cambios:
            filas.append({"columna": f"{a}_cambio", "variable_origen": a,
                          "año": None, "rol": "indicador de cambio"})
    for c in med.columns:
        var, _, anio = c.rpartition("_")
        filas.append({"columna": c, "variable_origen": var,
                      "año": int(anio), "rol": "medida anual"})
    filas.append({"columna": "n_anios_presente", "variable_origen": col_anio,
                  "año": None, "rol": "derivada"})
    mapa = pd.DataFrame(filas)

    log.info("Panel ancho: %d establecimientos x %d columnas",
             len(ancho), ancho.shape[1])
    return ancho, mapa
