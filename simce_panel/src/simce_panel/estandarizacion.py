"""
estandarizacion.py
==================

Estandarización de las variables categóricas del panel SIMCE: convierte los
**códigos numéricos a su etiqueta de texto** usando el catálogo del **año
correspondiente**, de modo que toda la serie 2015-2025 quede con una regla
unificada.

Principio de diseño: la conversión es *year-aware*
--------------------------------------------------
No existe un diccionario global válido para toda la serie (ver
:mod:`glosarios`): ``cod_depe1`` gana una categoría en 2018 y, sobre todo, en
``marcadif_*`` los códigos 1 y 2 **intercambian significado en 2023**. Por eso
:func:`estandarizar_categoricas` traduce fila a fila según el año de la fila.

La función es **idempotente y tolerante al formato mixto**: si un año ya trae
la etiqueta en texto, la reconoce y la deja como está (normalizando la
redacción a la versión canónica); si trae el código (``1``, ``"1.0"``), lo
traduce. Los valores que no se pueden resolver se registran en un reporte en
lugar de perderse en silencio.

Funciones principales
---------------------
- :func:`concepto_de_variable`     : a qué catálogo pertenece una variable.
- :func:`estandarizar_categoricas` : traduce todas las categóricas del panel.
- :func:`validar_catalogos_contra_datos` : contrasta códigos observados vs catálogo.
"""

from __future__ import annotations

import logging

import pandas as pd

from .columnas import normalizar_texto
from .glosarios import (PREFIJOS_A_CONCEPTO, VARIABLE_A_CONCEPTO, _norm,
                        _normaliza_codigo, _similitud)

log = logging.getLogger(__name__)

#: Sufijo que se añade a la variable original al crear la versión en texto.
SUFIJO_ETIQUETA: str = "_desc"


def concepto_de_variable(variable: str) -> str | None:
    """Determina a qué catálogo pertenece una variable del panel.

    Reconoce tanto nombres exactos (``cod_depe1``) como prefijos por asignatura
    (``marca_mate2m_rbd`` → ``marca``; ``marcadif_lect2m_rbd`` → ``marcadif``).

    Parameters
    ----------
    variable : str

    Returns
    -------
    str | None
        Nombre del concepto, o ``None`` si la variable no es categórica.

    Examples
    --------
    >>> concepto_de_variable("cod_depe2")
    'dependencia2'
    >>> concepto_de_variable("marcadif_mate2m_rbd")
    'marcadif'
    """
    # Las columnas derivadas por el propio pipeline ('<var>_desc', '<var>_cat')
    # NO son categóricas de origen: volver a traducirlas produciría valores
    # irresolubles, porque su contenido ya es la etiqueta final.
    if isinstance(variable, str) and (variable.endswith("_desc")
                                      or variable.endswith("_cat")):
        return None

    v = normalizar_texto(variable)
    for nombre, concepto in VARIABLE_A_CONCEPTO.items():
        if v == normalizar_texto(nombre):
            return concepto
    for prefijo, concepto in PREFIJOS_A_CONCEPTO:
        if v.startswith(normalizar_texto(prefijo)):
            return concepto
    return None


def estandarizar_canonico(
    df: pd.DataFrame,
    *,
    col_anio: str = "año",
    variables: list[str] | None = None,
    sufijo: str = "_cat",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Añade la **categoría canónica armonizada** de cada variable categórica.

    Usa :mod:`equivalencias`, que resuelve tanto códigos como los distintos
    vocabularios de texto observados en los datos, y armoniza los esquemas que
    cambiaron entre años (permutación y fusión de categorías en ``marcadif``).

    A diferencia de :func:`estandarizar_categoricas` (que reproduce la etiqueta
    oficial del año, fiel a la fuente), esta función produce la columna que se
    debe usar para **comparar o modelar entre años**.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo.
    col_anio : str, optional
    variables : list[str] | None, optional
        Si es ``None``, se autodetectan las categóricas.
    sufijo : str, optional
        Sufijo de la columna creada (por defecto ``_cat``).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(df_con_categorias, reporte)``; el reporte indica cuántos valores se
        resolvieron y cuáles quedaron pendientes por variable y año.
    """
    from .equivalencias import categoria_canonica

    out = df.copy()
    if variables is None:
        variables = [c for c in out.columns if concepto_de_variable(c)]
    anios = pd.to_numeric(out[col_anio], errors="coerce")
    filas = []
    for var in variables:
        concepto = concepto_de_variable(var)
        destino = f"{var}{sufijo}"
        out[destino] = [
            categoria_canonica(v, concepto, a if pd.notna(a) else None)
            for v, a in zip(out[var], anios)
        ]
        for anio in sorted(anios.dropna().astype(int).unique()):
            m = anios == anio
            crudo = out.loc[m, var]
            no_vacio = crudo.notna() & (crudo.astype(str).str.strip() != "")
            resuelto = out.loc[m, destino].notna()
            pend = crudo[no_vacio & ~resuelto]
            filas.append({
                "variable": var, "concepto": concepto, "año": anio,
                "n_con_valor": int(no_vacio.sum()),
                "n_resuelto": int((no_vacio & resuelto).sum()),
                "n_pendiente": int(len(pend)),
                "ejemplos_pendientes": sorted({str(x) for x in pend.unique()})[:4],
            })
    reporte = pd.DataFrame(filas)
    pend_total = int(reporte["n_pendiente"].sum()) if not reporte.empty else 0
    if pend_total:
        log.warning("%d valores sin categoría canónica; revisa el reporte.", pend_total)
    return out, reporte


def _resolver_valor(
    valor: object,
    tabla: dict[str, str],
    umbral_texto: float = 0.85,
) -> tuple[str | None, str]:
    """Resuelve un valor a su etiqueta canónica.

    Returns
    -------
    tuple[str | None, str]
        ``(etiqueta, estado)`` donde estado es ``"vacio"``, ``"codigo"``
        (traducido desde el código), ``"texto"`` (ya venía como texto y se
        normalizó a la redacción canónica) o ``"no_resuelto"``.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None, "vacio"
    bruto = str(valor).strip()
    if bruto == "" or bruto.lower() in {"nan", "none", "<na>"}:
        return None, "vacio"

    # 1) ¿Es un código presente en la tabla del año?
    cod = _normaliza_codigo(bruto)
    if cod in tabla:
        return tabla[cod], "codigo"

    # 2) ¿Ya es texto? Se empareja con la etiqueta canónica más parecida.
    mejor, mejor_sim = None, 0.0
    for etiqueta in tabla.values():
        s = _similitud(bruto, etiqueta)
        if s > mejor_sim:
            mejor, mejor_sim = etiqueta, s
    if mejor is not None and mejor_sim >= umbral_texto:
        return mejor, "texto"

    return None, "no_resuelto"


def estandarizar_categoricas(
    df: pd.DataFrame,
    catalogos: dict[tuple[str, int], dict[str, str]],
    *,
    col_anio: str = "año",
    variables: list[str] | None = None,
    reemplazar: bool = False,
    sufijo: str = SUFIJO_ETIQUETA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Traduce las variables categóricas a texto usando el catálogo de cada año.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel largo (una fila por establecimiento-año) con la columna de año.
    catalogos : dict[tuple[str, int], dict[str, str]]
        Salida de :func:`glosarios.construir_catalogos`.
    col_anio : str, optional
        Nombre de la columna de año.
    variables : list[str] | None, optional
        Variables a estandarizar. Si es ``None``, se autodetectan con
        :func:`concepto_de_variable`.
    reemplazar : bool, optional
        Si ``True`` sobrescribe la columna original con la etiqueta; si es
        ``False`` (por defecto) crea una columna nueva ``<variable><sufijo>`` y
        conserva el código original (recomendado: preserva la trazabilidad).
    sufijo : str, optional

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(df_estandarizado, reporte)``. El reporte trae una fila por variable
        y año con los conteos por estado y ejemplos de valores no resueltos.
    """
    out = df.copy()
    if variables is None:
        variables = [c for c in out.columns if concepto_de_variable(c)]

    anios = sorted(pd.to_numeric(out[col_anio], errors="coerce")
                   .dropna().astype(int).unique().tolist())
    filas_reporte = []

    for var in variables:
        concepto = concepto_de_variable(var)
        if concepto is None:
            continue
        destino = var if reemplazar else f"{var}{sufijo}"
        if destino not in out.columns:
            out[destino] = pd.Series([pd.NA] * len(out), dtype="object")

        for anio in anios:
            tabla = catalogos.get((concepto, anio))
            mask = pd.to_numeric(out[col_anio], errors="coerce") == anio
            if tabla is None:
                log.warning("Sin catálogo para '%s' en %s; %d filas sin traducir.",
                            concepto, anio, int(mask.sum()))
                filas_reporte.append({
                    "variable": var, "concepto": concepto, "año": anio,
                    "n_filas": int(mask.sum()), "n_desde_codigo": 0,
                    "n_desde_texto": 0, "n_vacio": 0,
                    "n_no_resuelto": int(mask.sum()),
                    "ejemplos_no_resueltos": ["(sin catálogo)"],
                })
                continue

            sub = out.loc[mask, var]
            resueltos, estados = [], []
            for v in sub:
                et, est = _resolver_valor(v, tabla)
                resueltos.append(et)
                estados.append(est)
            out.loc[mask, destino] = resueltos

            est_ser = pd.Series(estados, index=sub.index)
            no_res = sub[est_ser == "no_resuelto"]
            filas_reporte.append({
                "variable": var, "concepto": concepto, "año": anio,
                "n_filas": int(mask.sum()),
                "n_desde_codigo": int((est_ser == "codigo").sum()),
                "n_desde_texto": int((est_ser == "texto").sum()),
                "n_vacio": int((est_ser == "vacio").sum()),
                "n_no_resuelto": int(len(no_res)),
                "ejemplos_no_resueltos": sorted(
                    {str(x) for x in no_res.unique()})[:4],
            })

    reporte = pd.DataFrame(filas_reporte)
    if not reporte.empty and reporte["n_no_resuelto"].sum():
        log.warning("Quedaron %d valores sin resolver; revisa el reporte.",
                    int(reporte["n_no_resuelto"].sum()))
    return out, reporte


def validar_catalogos_contra_datos(
    df: pd.DataFrame,
    catalogos: dict[tuple[str, int], dict[str, str]],
    *,
    col_anio: str = "año",
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Contrasta los códigos realmente presentes en los datos con el catálogo.

    Sirve para **verificar los catálogos supuestos** de 2015-2018: si en esos
    años aparece un código que el catálogo supuesto no contempla (por ejemplo,
    un 6 en ``cod_depe1`` antes de 2018), el supuesto es incorrecto.

    Parameters
    ----------
    df : pandas.DataFrame
    catalogos : dict[tuple[str, int], dict[str, str]]
    col_anio : str, optional
    variables : list[str] | None, optional

    Returns
    -------
    pandas.DataFrame
        Por variable y año: códigos observados, códigos del catálogo, y los
        que sobran o faltan.
    """
    if variables is None:
        variables = [c for c in df.columns if concepto_de_variable(c)]
    anios = sorted(pd.to_numeric(df[col_anio], errors="coerce")
                   .dropna().astype(int).unique().tolist())
    filas = []
    for var in variables:
        concepto = concepto_de_variable(var)
        for anio in anios:
            mask = pd.to_numeric(df[col_anio], errors="coerce") == anio
            obs = {_normaliza_codigo(v) for v in df.loc[mask, var].dropna().unique()}
            obs = {o for o in obs if o != ""}
            # Solo interesan los valores que parecen códigos (cortos/numéricos).
            obs_cod = {o for o in obs if len(o) <= 3}
            tabla = catalogos.get((concepto, anio), {})
            filas.append({
                "variable": var, "concepto": concepto, "año": anio,
                "codigos_observados": sorted(obs_cod),
                "codigos_en_catalogo": sorted(tabla.keys()),
                "observados_fuera_de_catalogo": sorted(obs_cod - set(tabla)),
                "catalogo_sin_uso": sorted(set(tabla) - obs_cod),
            })
    return pd.DataFrame(filas)
