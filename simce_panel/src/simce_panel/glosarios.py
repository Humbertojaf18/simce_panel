"""
glosarios.py
============

Lectura de los glosarios oficiales (``*_GLOSAS_*.xlsx``) para extraer, **por
año**, dos cosas:

1. Las **descripciones** de cada variable (hoja del nivel correspondiente).
2. Los **catálogos de códigos** (tablas "Valor / Descripción" al pie de la
   hoja): dependencia, GSE, ruralidad, marcas, significancia y no-aplica.

Por qué por año y no un diccionario global
------------------------------------------
Los catálogos NO son estables en la serie 2015-2025:

- ``cod_depe1`` tiene **5 categorías hasta 2017 y 6 desde 2018** (se agrega
  "Servicio Local de Educación"); ``cod_depe2`` pasa de 3 a 4 categorías.
- En ``marcadif_*`` los códigos **1 y 2 intercambian significado entre 2022 y
  2023**: en 2018-2022 el 1 es "cantidad de estudiantes insuficiente", mientras
  que en 2023-2025 el 1 es "no hubo resultados / comparación no válida" y el 2
  pasa a ser el de cantidad insuficiente.

Aplicar un único diccionario a toda la serie produciría etiquetas
sistemáticamente equivocadas. Por eso este módulo entrega catálogos indexados
por ``(concepto, año)``.

Los glosarios de 2015-2017 no incluyen tablas de códigos (y el de 2018 solo
trae marcas y no-aplica); para esos años se usa
:data:`CATALOGOS_SUPUESTOS`, explícitamente marcado como supuesto verificable.

Funciones principales
---------------------
- :func:`mapear_glosarios`     : {año: ruta} de los archivos de glosas.
- :func:`leer_descripciones`   : {variable: descripción} de un año.
- :func:`leer_catalogos_anio`  : {concepto: {código: etiqueta}} de un año.
- :func:`construir_catalogos`  : {(concepto, año): {código: etiqueta}} completo.
- :func:`catalogos_a_dataframe`: los catálogos como tabla auditable.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import pandas as pd

from .io_datos import extraer_anio

log = logging.getLogger(__name__)

#: Conceptos categóricos y los títulos con que aparecen en los glosarios.
#: La comparación es por texto normalizado y por "contiene".
TITULOS_CATALOGO: dict[str, tuple[str, ...]] = {
    "dependencia1": ("dependencia 1",),
    "dependencia2": ("dependencia 2",),
    "gse": ("gse",),
    "ruralidad": ("ruralidad",),
    "marca": ("marca: observaciones al puntaje",),
    "marcadif": ("marca: observaciones a las diferencias del puntaje",),
    "significancia": ("significancia",),
    "noaplica": ("no aplica",),
}

#: Qué concepto aplica a cada variable del panel (por prefijo del nombre).
#: Se resuelve por coincidencia de prefijo sobre el nombre normalizado.
VARIABLE_A_CONCEPTO: dict[str, str] = {
    "cod_depe1": "dependencia1",
    "cod_depe2": "dependencia2",
    "cod_grupo": "gse",
    "cod_rural_rbd": "ruralidad",
    "noaplica": "noaplica",
}

#: Prefijos para variables por asignatura (marca_*, marcadif_*, sigdif_*, ...).
PREFIJOS_A_CONCEPTO: tuple[tuple[str, str], ...] = (
    ("marcapalu_", "marcapalu"),    # concepto propio (umbral N<10)
    ("marcadif_", "marcadif"),      # antes que 'marca_' (más específico)
    ("marca_", "marca"),
    ("sigdif_", "significancia"),
    ("siggru_", "significancia"),
)

#: Catálogos supuestos para los años cuyo glosario no trae la tabla
#: (2015-2017 y, parcialmente, 2018). Se derivan de la versión vigente
#: restringida al número de categorías que declara el glosario de esos años
#: ("5 categorías" en depe1 y "3 categorías" en depe2), es decir, sin la
#: categoría "Servicio Local de Educación", que no existía antes de 2018.
#: SUPUESTO VERIFICABLE: contrastar contra los códigos realmente presentes en
#: los CSV de esos años con :func:`validar_catalogos_contra_datos`.
CATALOGOS_SUPUESTOS: dict[str, dict[str, str]] = {
    "dependencia1": {
        "1": "Municipal Corporación",
        "2": "Municipal DAEM",
        "3": "Particular subvencionado",
        "4": "Particular pagado",
        "5": "Corporación de administración delegada",
    },
    "dependencia2": {
        "1": "Municipal",
        "2": "Particular subvencionado",
        "3": "Particular pagado",
    },
    "gse": {
        "1": "Bajo", "2": "Medio bajo", "3": "Medio",
        "4": "Medio alto", "5": "Alto",
    },
    "ruralidad": {"1": "Urbano", "2": "Rural"},
    "significancia": {
        "1": "Diferencia positiva y estadísticamente significativa",
        "0": "Diferencia estadísticamente NO significativa",
        "-1": "Diferencia negativa y estadísticamente significativa",
    },
    # Para 2015-2017 se adopta la tabla oficial más antigua disponible (2018),
    # que coincide con la de 2022. NO se usa la de 2023+, porque allí los
    # códigos 1 y 2 de 'marcadif' están permutados.
    "marca": {
        "1": "No es posible reportar resultados, porque la cantidad de "
             "estudiantes evaluados es insuficiente. (6 o menos)",
        "2": "Por causas ajenas a la Agencia, los resultados no son "
             "representativos del desempeño de los estudiantes",
        "3": "Por causas ajenas al establecimiento, los resultados no son "
             "representativos del desempeño de los estudiantes",
        "4": "La aplicación de la prueba extendida no permite evaluar esta "
             "asignatura.",
    },
    "marcadif": {
        "1": "No es posible comparar resultados, porque la cantidad de "
             "estudiantes evaluados es insuficiente. (6 o menos)",
        "2": "No es posible reportar esta información, porque en la evaluación "
             "anterior no hubo resultados o estos no permiten una comparación "
             "válida.",
        "3": "La comparación no es representativa, porque en la evaluación "
             "actual los resultados no permiten una comparación válida.",
    },
}


def _norm(s: object) -> str:
    """Normaliza texto para comparar títulos (minúsculas, sin acentos)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    txt = unicodedata.normalize("NFKD", str(s))
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return " ".join(txt.lower().split())


def mapear_glosarios(carpeta: str | Path) -> dict[int, Path]:
    """Busca los archivos de glosas y los indexa por año.

    Parameters
    ----------
    carpeta : str | Path
        Carpeta donde buscar recursivamente (p. ej. ``Dataset/Simce 2°medio``).

    Returns
    -------
    dict[int, Path]
    """
    carpeta = Path(carpeta)
    mapa: dict[int, Path] = {}
    for xlsx in sorted(carpeta.rglob("*.xlsx")):
        if "glosa" not in _norm(xlsx.stem):
            continue
        anio = extraer_anio(xlsx.name) or extraer_anio(xlsx.parent.name)
        if anio is not None and anio not in mapa:
            mapa[anio] = xlsx
    return mapa


def _hoja_del_nivel(xlsx: Path, nivel: str = "rbd") -> str:
    """Devuelve el nombre de la hoja correspondiente al nivel pedido."""
    hojas = pd.ExcelFile(xlsx).sheet_names
    for h in hojas:
        if nivel in _norm(h):
            return h
    raise KeyError(f"No hay hoja '{nivel}' en {xlsx.name}. Hojas: {hojas}")


def leer_descripciones(xlsx: str | Path, nivel: str = "rbd") -> dict[str, str]:
    """Extrae ``{variable: descripción}`` de la hoja de un glosario.

    Parameters
    ----------
    xlsx : str | Path
    nivel : str, optional
        ``rbd``, ``comuna``, ``deprov`` o ``region``.

    Returns
    -------
    dict[str, str]
    """
    xlsx = Path(xlsx)
    df = pd.read_excel(xlsx, sheet_name=_hoja_del_nivel(xlsx, nivel),
                       header=None, dtype=str)
    inicio = None
    for i in range(len(df)):
        if _norm(df.iat[i, 0]) == "nombre" and _norm(df.iat[i, 1]).startswith("descrip"):
            inicio = i + 1
            break
    if inicio is None:
        return {}
    out: dict[str, str] = {}
    for i in range(inicio, len(df)):
        nombre = str(df.iat[i, 0]).strip() if pd.notna(df.iat[i, 0]) else ""
        if not nombre:
            break
        desc = str(df.iat[i, 1]).strip() if pd.notna(df.iat[i, 1]) else ""
        out[nombre] = desc
    return out


def leer_catalogos_anio(xlsx: str | Path, nivel: str = "rbd") -> dict[str, dict[str, str]]:
    """Extrae los catálogos de códigos ("Valor / Descripción") de un glosario.

    Recorre la hoja buscando cabeceras ``Valor | Descripción``; el título del
    catálogo es la última celda no vacía de la columna A por encima de ella.

    Parameters
    ----------
    xlsx : str | Path
    nivel : str, optional

    Returns
    -------
    dict[str, dict[str, str]]
        ``{concepto: {código: etiqueta}}`` solo para los conceptos reconocidos
        en :data:`TITULOS_CATALOGO`.
    """
    xlsx = Path(xlsx)
    df = pd.read_excel(xlsx, sheet_name=_hoja_del_nivel(xlsx, nivel),
                       header=None, dtype=str)
    n = len(df)
    encontrados: dict[str, dict[str, str]] = {}

    for i in range(n):
        if _norm(df.iat[i, 0]) != "valor":
            continue
        if not _norm(df.iat[i, 1]).startswith("descrip"):
            continue
        # Título: primera celda no vacía hacia arriba en la columna A.
        titulo = ""
        for k in range(i - 1, max(-1, i - 5), -1):
            cand = str(df.iat[k, 0]).strip() if pd.notna(df.iat[k, 0]) else ""
            if cand:
                titulo = cand
                break
        concepto = None
        tnorm = _norm(titulo)
        for c, claves in TITULOS_CATALOGO.items():
            if any(k in tnorm for k in claves):
                concepto = c
                break
        if concepto is None:
            continue
        # Pares hasta la primera fila completamente vacía.
        pares: dict[str, str] = {}
        j = i + 1
        while j < n:
            va = str(df.iat[j, 0]).strip() if pd.notna(df.iat[j, 0]) else ""
            vb = str(df.iat[j, 1]).strip() if pd.notna(df.iat[j, 1]) else ""
            if not va and not vb:
                break
            if va:
                pares[_normaliza_codigo(va)] = vb
            j += 1
        if pares:
            encontrados.setdefault(concepto, pares)
    return encontrados


def _normaliza_codigo(v: object) -> str:
    """Normaliza un código a texto canónico: ``'1.0'`` y ``1`` -> ``'1'``."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s == "":
        return ""
    try:
        f = float(s.replace(",", "."))
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return s


def construir_catalogos(
    carpeta_simce: str | Path,
    nivel: str = "rbd",
    *,
    usar_supuestos: bool = True,
) -> dict[tuple[str, int], dict[str, str]]:
    """Construye los catálogos indexados por ``(concepto, año)``.

    Lee cada glosario disponible; para los años cuyo glosario no trae una tabla
    de un concepto, completa con :data:`CATALOGOS_SUPUESTOS` (si
    ``usar_supuestos``), registrando una advertencia.

    Parameters
    ----------
    carpeta_simce : str | Path
        Carpeta con las subcarpetas por año (contiene los ``*GLOSAS*.xlsx``).
    nivel : str, optional
    usar_supuestos : bool, optional

    Returns
    -------
    dict[tuple[str, int], dict[str, str]]
    """
    mapa = mapear_glosarios(carpeta_simce)
    catalogos: dict[tuple[str, int], dict[str, str]] = {}
    conceptos_vistos: set[str] = set()

    for anio, xlsx in sorted(mapa.items()):
        try:
            cats = leer_catalogos_anio(xlsx, nivel)
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo leer catálogos de %s: %s", xlsx.name, exc)
            cats = {}
        for concepto, tabla in cats.items():
            catalogos[(concepto, anio)] = tabla
            conceptos_vistos.add(concepto)
        log.info("Glosario %s: catálogos leídos -> %s",
                 anio, sorted(cats.keys()) or "(ninguno)")

    if usar_supuestos:
        for anio in sorted(mapa):
            for concepto, tabla in CATALOGOS_SUPUESTOS.items():
                if (concepto, anio) not in catalogos:
                    catalogos[(concepto, anio)] = dict(tabla)
                    log.warning(
                        "Año %s: sin tabla oficial para '%s'; se usa catálogo "
                        "SUPUESTO (verificar contra los datos).", anio, concepto)
    return catalogos


def catalogos_a_dataframe(
    catalogos: dict[tuple[str, int], dict[str, str]],
) -> pd.DataFrame:
    """Convierte los catálogos en una tabla auditable.

    Returns
    -------
    pandas.DataFrame
        Columnas: ``concepto``, ``año``, ``codigo``, ``etiqueta``.
    """
    filas = []
    for (concepto, anio), tabla in sorted(catalogos.items()):
        for codigo, etiqueta in tabla.items():
            filas.append({"concepto": concepto, "año": anio,
                          "codigo": codigo, "etiqueta": etiqueta})
    return pd.DataFrame(filas)


#: Umbral de similitud por encima del cual dos etiquetas se consideran la
#: misma categoría redactada de otro modo (corrección de erratas, reformulación).
UMBRAL_SIMILITUD: float = 0.75


def _similitud(a: str, b: str) -> float:
    """Similitud entre dos etiquetas normalizadas (0 a 1)."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def detectar_cambios_de_catalogo(
    catalogos: dict[tuple[str, int], dict[str, str]],
    umbral: float = UMBRAL_SIMILITUD,
) -> pd.DataFrame:
    """Detecta años en que un código cambia de significado o aparece/desaparece.

    Es la evidencia de por qué la conversión debe hacerse **por año**. Distingue
    tres situaciones, para no confundir una errata corregida con una ruptura:

    - ``código nuevo`` / ``código eliminado``: el código no existía o deja de existir.
    - ``cambio de redacción``: la etiqueta cambia pero describe lo mismo
      (similitud ≥ ``umbral``); no rompe la serie.
    - ``CAMBIO DE SIGNIFICADO``: la etiqueta pasa a designar otra categoría
      (similitud < ``umbral``). Este es el caso peligroso.

    Parameters
    ----------
    catalogos : dict[tuple[str, int], dict[str, str]]
    umbral : float, optional
        Similitud mínima para considerar que es solo redacción.

    Returns
    -------
    pandas.DataFrame
        Columnas: ``concepto``, ``codigo``, ``año_desde``, ``etiqueta_anterior``,
        ``etiqueta_nueva``, ``similitud``, ``tipo_cambio``.
    """
    filas = []
    conceptos = sorted({c for c, _ in catalogos})
    for concepto in conceptos:
        anios = sorted(a for c, a in catalogos if c == concepto)
        for prev, act in zip(anios, anios[1:]):
            ta, tb = catalogos[(concepto, prev)], catalogos[(concepto, act)]
            for cod in sorted(set(ta) | set(tb)):
                ea, eb = ta.get(cod), tb.get(cod)
                if ea == eb:
                    continue
                if ea is None:
                    tipo, sim = "código nuevo", None
                elif eb is None:
                    tipo, sim = "código eliminado", None
                else:
                    sim = round(_similitud(ea, eb), 3)
                    if sim >= umbral:
                        tipo = "cambio de redacción"
                    else:
                        # ¿La etiqueta vieja de este código reaparece en OTRO
                        # código del año nuevo? Entonces los códigos se
                        # permutaron: el error más grave posible.
                        permutado = any(
                            otro != cod and _similitud(ea, et) >= umbral
                            for otro, et in tb.items()
                        )
                        tipo = ("PERMUTACIÓN DE CÓDIGOS" if permutado
                                else "CAMBIO DE SIGNIFICADO")
                filas.append({
                    "concepto": concepto, "codigo": cod, "año_desde": act,
                    "etiqueta_anterior": ea, "etiqueta_nueva": eb,
                    "similitud": sim, "tipo_cambio": tipo,
                })
    return pd.DataFrame(filas)
