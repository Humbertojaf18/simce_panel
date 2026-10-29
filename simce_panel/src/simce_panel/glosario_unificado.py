"""
glosario_unificado.py
=====================

Generación del **glosario consolidado** de la base unificada en formato ancho.

A diferencia de :mod:`glosario` (que documenta el panel largo), aquí se
documenta la tabla final de una fila por establecimiento, cruzando tres
fuentes de información:

1. El **mapa de columnas** producido por :func:`panel_ancho.construir_panel_ancho`
   (qué variable de origen y qué año representa cada columna).
2. Las **descripciones oficiales** de los glosarios por año (:mod:`glosarios`).
3. Los **catálogos de valores** vigentes por año, para las variables
   categóricas, incluyendo la advertencia cuando un código cambió de
   significado en la serie.

El resultado es un Excel con cuatro hojas:

- ``DICCIONARIO``  : una fila por columna del panel ancho.
- ``VALORES``      : catálogo de valores estandarizados por concepto y año.
- ``CAMBIOS``      : rupturas detectadas en los catálogos (auditoría).
- ``RESUMEN``      : metadatos de la base (filas, columnas, años, cobertura).

Funciones principales
---------------------
- :func:`construir_diccionario` : DataFrame con una fila por columna final.
- :func:`exportar_glosario_unificado` : escribe el Excel de cuatro hojas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .estandarizacion import concepto_de_variable
from .glosarios import catalogos_a_dataframe, detectar_cambios_de_catalogo

log = logging.getLogger(__name__)

#: Descripciones propias de las columnas que crea el pipeline.
DESCRIPCIONES_DERIVADAS: dict[str, str] = {
    "n_anios_presente": "Número de años (de la serie 2015-2025) en que el "
                        "establecimiento aparece con datos en el panel.",
    "latitud": "Latitud del establecimiento (grados decimales, WGS84), "
               "anexada desde el Directorio Oficial.",
    "longitud": "Longitud del establecimiento (grados decimales, WGS84), "
                "anexada desde el Directorio Oficial.",
}

#: Plantilla de descripción para los indicadores de cambio de atributo.
DESC_CAMBIO = ("Indica si '{var}' tomó más de un valor distinto a lo largo de "
               "la serie (por ejemplo, traspaso de sostenedor o cambio "
               "administrativo). El valor de ficha corresponde al año más "
               "reciente disponible.")


def construir_diccionario(
    panel_ancho: pd.DataFrame,
    mapa_columnas: pd.DataFrame,
    descripciones: dict[str, str],
    catalogos: dict[tuple[str, int], dict[str, str]],
) -> pd.DataFrame:
    """Construye el diccionario de datos del panel ancho.

    Parameters
    ----------
    panel_ancho : pandas.DataFrame
        Tabla final (una fila por RBD).
    mapa_columnas : pandas.DataFrame
        Mapa columna → (variable de origen, año, rol).
    descripciones : dict[str, str]
        ``{variable: descripción}`` consolidado desde los glosarios oficiales.
    catalogos : dict[tuple[str, int], dict[str, str]]

    Returns
    -------
    pandas.DataFrame
        Columnas: ``columna``, ``variable_origen``, ``año``, ``rol``,
        ``descripcion``, ``tipo_dato``, ``valores_estandarizados``,
        ``no_nulos``, ``pct_completitud``, ``advertencia``.
    """
    n = len(panel_ancho)
    # Conceptos cuyo catálogo cambió de forma no trivial.
    cambios = detectar_cambios_de_catalogo(catalogos)
    criticos = set()
    if not cambios.empty:
        criticos = set(cambios.loc[
            cambios["tipo_cambio"].isin(
                ["PERMUTACIÓN DE CÓDIGOS", "CAMBIO DE SIGNIFICADO"]),
            "concepto"])

    filas = []
    for _, r in mapa_columnas.iterrows():
        col = r["columna"]
        if col not in panel_ancho.columns:
            continue
        var = r["variable_origen"]
        rol = r["rol"]

        # Descripción
        if rol == "indicador de cambio":
            desc = DESC_CAMBIO.format(var=var)
        elif col in DESCRIPCIONES_DERIVADAS:
            desc = DESCRIPCIONES_DERIVADAS[col]
        elif var in DESCRIPCIONES_DERIVADAS:
            desc = DESCRIPCIONES_DERIVADAS[var]
        else:
            # Las columnas '<var>_desc' son la etiqueta de texto estandarizada
            # de '<var>': heredan su descripción y lo declaran explícitamente.
            var_base, es_desc = var, False
            if isinstance(var, str) and var.endswith("_desc"):
                var_base, es_desc = var[:-5], True
            base = descripciones.get(var_base, "")
            if es_desc:
                base = ((base + " ") if base else "") + (
                    "Etiqueta de texto estandarizada a partir del código, "
                    "usando el catálogo del año correspondiente.")
            desc = (f"{base} (año {int(r['año'])})"
                    if base and pd.notna(r["año"]) else base)

        # Valores estandarizados (para categóricas ya traducidas a texto)
        # Para '<var>_desc' / '<var>_cat' el catálogo es el de la variable base.
        var_cat = var
        if isinstance(var, str):
            for suf in ("_desc", "_cat"):
                if var.endswith(suf):
                    var_cat = var[: -len(suf)]
                    break
        concepto = concepto_de_variable(var_cat) if isinstance(var_cat, str) else None
        valores, advertencia = "", ""
        if concepto:
            anio = int(r["año"]) if pd.notna(r["año"]) else None
            tabla = None
            if anio is not None:
                tabla = catalogos.get((concepto, anio))
            if tabla is None:  # atributo de ficha: catálogo más reciente
                disponibles = sorted(a for c, a in catalogos if c == concepto)
                if disponibles:
                    tabla = catalogos[(concepto, disponibles[-1])]
            if tabla:
                valores = " | ".join(sorted(set(tabla.values())))
            if concepto in criticos:
                advertencia = ("El catálogo de este concepto cambió durante la "
                               "serie; la traducción se aplicó con el catálogo "
                               "del año de cada dato (ver hoja CAMBIOS).")

        no_nulos = int(panel_ancho[col].notna().sum())
        filas.append({
            "columna": col,
            "variable_origen": var,
            "año": (int(r["año"]) if pd.notna(r["año"]) else None),
            "rol": rol,
            "descripcion": desc,
            "tipo_dato": str(panel_ancho[col].dtype),
            "valores_estandarizados": valores,
            "no_nulos": no_nulos,
            "pct_completitud": round(100 * no_nulos / n, 2) if n else 0.0,
            "advertencia": advertencia,
        })
    return pd.DataFrame(filas)


def exportar_glosario_unificado(
    panel_ancho: pd.DataFrame,
    mapa_columnas: pd.DataFrame,
    descripciones: dict[str, str],
    catalogos: dict[tuple[str, int], dict[str, str]],
    ruta_salida: str | Path,
) -> Path:
    """Genera el Excel del glosario consolidado (4 hojas).

    Parameters
    ----------
    panel_ancho : pandas.DataFrame
    mapa_columnas : pandas.DataFrame
    descripciones : dict[str, str]
    catalogos : dict[tuple[str, int], dict[str, str]]
    ruta_salida : str | Path

    Returns
    -------
    Path
        Ruta del archivo generado.
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    dicc = construir_diccionario(panel_ancho, mapa_columnas,
                                 descripciones, catalogos)
    valores = catalogos_a_dataframe(catalogos)
    cambios = detectar_cambios_de_catalogo(catalogos)

    anios = sorted({int(a) for a in mapa_columnas["año"].dropna().unique()})
    resumen = pd.DataFrame([
        {"metadato": "Unidad de análisis", "valor": "Establecimiento (RBD) — una fila por establecimiento"},
        {"metadato": "Establecimientos (filas)", "valor": len(panel_ancho)},
        {"metadato": "Columnas", "valor": panel_ancho.shape[1]},
        {"metadato": "Años cubiertos", "valor": ", ".join(map(str, anios))},
        {"metadato": "Formato de columnas anuales", "valor": "<variable>_<año>"},
        {"metadato": "Atributos de ficha", "valor": "Valor del año más reciente; '<atributo>_cambio' indica variación en la serie"},
        {"metadato": "Categóricas", "valor": "Traducidas a texto con el catálogo del año de cada dato"},
        {"metadato": "Conceptos con ruptura de catálogo", "valor": ", ".join(sorted(
            set(cambios.loc[cambios["tipo_cambio"].isin(
                ["PERMUTACIÓN DE CÓDIGOS", "CAMBIO DE SIGNIFICADO"]), "concepto"]))) or "ninguno"},
    ])

    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        dicc.to_excel(w, sheet_name="DICCIONARIO", index=False)
        valores.to_excel(w, sheet_name="VALORES", index=False)
        (cambios if not cambios.empty else pd.DataFrame(
            columns=["concepto", "codigo", "año_desde", "tipo_cambio"])
         ).to_excel(w, sheet_name="CAMBIOS", index=False)
        resumen.to_excel(w, sheet_name="RESUMEN", index=False)

    log.info("Glosario consolidado exportado: %s", ruta)
    return ruta
