"""
equivalencias.py
================

Tabla de equivalencias construida a partir de los **valores realmente
observados** en los CSV de 2015-2025, no de supuestos.

Problema que resuelve
---------------------
Las variables categóricas del SIMCE llegan en tres formatos distintos según el
año y la variable:

- **Código numérico** (2018-2025, y ``marca_*`` ya en 2015).
- **Texto, vocabulario A** (2015): ``"Evaluación actual no permite comparar"``.
- **Texto, vocabulario B** (2016-2017): ``"Con marca en esta medición: No representativa"``.

Además, el emparejamiento difuso por similitud **no sirve** aquí: probado
contra los datos reales, ``"Corporación Municipal"`` se parece más a
``"Corporación de administración delegada"`` (0,61) que a su categoría correcta
``"Municipal Corporación"``, y los textos de ``marcadif_*`` no superan 0,46 de
similitud con la redacción oficial. Por eso las equivalencias son **explícitas**.

Las dos capas de salida
-----------------------
1. ``<variable>_desc`` — **etiqueta oficial del año**. Máxima fidelidad a la
   fuente; sirve para reportar y auditar.
2. ``<variable>_cat`` — **categoría canónica armonizada**, estable en toda la
   serie. Es la que se debe usar para comparar o modelar entre años.

La segunda capa es necesaria porque en 2023 la Agencia no solo permutó los
códigos de ``marcadif_*``: **fusionó** las categorías "la evaluación anterior
no permite comparar" y "la evaluación actual no permite comparar" en una sola.
Ninguna traducción código→texto puede deshacer esa fusión; lo correcto es
armonizar al nivel más grueso que sea comparable en todos los años.

Contenido
---------
- :data:`ALIAS_TEXTO`    : {concepto: {texto_normalizado: código canónico}}.
- :data:`CATEGORIAS`     : {concepto: {código canónico: categoría armonizada}}.
- :func:`codigo_canonico`: resuelve un valor crudo (código o texto) a su código.
- :func:`categoria_canonica` : resuelve un valor crudo a su categoría armonizada.
"""

from __future__ import annotations


#: Alias de texto observados en los datos → código canónico del concepto.
#: Las claves se comparan normalizadas (minúsculas, sin acentos ni signos).
ALIAS_TEXTO: dict[str, dict[str, str]] = {
    "dependencia1": {
        "corporacion municipal": "1",          # 2015-2017
        "municipal corporacion": "1",          # redacción del glosario
        "municipal daem": "2",
        "particular subvencionado": "3",
        "particular pagado": "4",
        "administracion delegada": "5",        # 2015-2017
        "corporacion de administracion delegada": "5",
        "servicio local de educacion": "6",
    },
    "dependencia2": {
        "municipal": "1",
        "particular subvencionado": "2",
        "particular pagado": "3",
        "servicio local de educacion": "4",
    },
    "gse": {
        "bajo": "1", "medio bajo": "2", "medio": "3",
        "medio alto": "4", "alto": "5",
    },
    "ruralidad": {"urbano": "1", "rural": "2"},
    "significancia": {
        "positiva y significativa": "1",
        "no significativa": "0",
        "negativa y significativa": "-1",
    },
    "marca": {
        "alumnos insuficientes": "1",
        "ajenas a la agencia resultados no representativos": "2",
        "ajenas al establecimiento resultados no representativos": "3",
    },
    # Vocabularios A (2015) y B (2016-2017) del mismo concepto.
    "marcadif": {
        "alumnos insuficientes": "1",
        "alumnos insuficientes para comparacion": "1",
        "evaluacion anterior no permite comparar": "2",
        "con marca medicion anterior no reporte": "2",
        "evaluacion actual no permite comparar": "3",
        "con marca en esta medicion no representativa": "3",
    },
    "marcapalu": {
        "alumnos insuficientes n 10": "1",
    },
    "noaplica": {
        "rbd y curso rinde simce": "0",
        "cursos con 100 ausentes y sin cuestionarios": "1",
        "cursos con 100 alumnos pie": "2",
        "cursos con 100 ausentes y con algun cuestionario": "3",
        "cursos con 100 alumnos que se niegan a responder": "4",
        "cursos en paro": "5",
        "cursos con material perdido en la aplicacion": "6",
    },
}

#: Categorías canónicas armonizadas, estables en toda la serie 2015-2025.
#: Clave externa: concepto. Clave interna: código canónico (el de 2018-2022,
#: que es el esquema más detallado y el que cubre más años).
CATEGORIAS: dict[str, dict[str, str]] = {
    "dependencia1": {
        "1": "Municipal Corporación",
        "2": "Municipal DAEM",
        "3": "Particular subvencionado",
        "4": "Particular pagado",
        "5": "Corporación de administración delegada",
        "6": "Servicio Local de Educación",
    },
    "dependencia2": {
        "1": "Municipal",
        "2": "Particular subvencionado",
        "3": "Particular pagado",
        "4": "Servicio Local de Educación",
    },
    "gse": {"1": "Bajo", "2": "Medio bajo", "3": "Medio",
            "4": "Medio alto", "5": "Alto"},
    "ruralidad": {"1": "Urbano", "2": "Rural"},
    "significancia": {
        "1": "Positiva y significativa",
        "0": "No significativa",
        "-1": "Negativa y significativa",
    },
    "marca": {
        "1": "Estudiantes insuficientes",
        "2": "No representativo (causas ajenas a la Agencia)",
        "3": "No representativo (causas ajenas al establecimiento)",
        "4": "Prueba extendida no evalúa la asignatura",
    },
    # ATENCIÓN: en 2023 la Agencia fusionó "evaluación anterior" y "evaluación
    # actual" en una sola categoría. Para que la variable sea comparable en
    # toda la serie, los códigos 2 y 3 del esquema antiguo se armonizan a una
    # única categoría "Comparación no válida". El detalle original se conserva
    # en la columna '<variable>_desc'.
    "marcadif": {
        "1": "Estudiantes insuficientes",
        "2": "Comparación no válida",
        "3": "Comparación no válida",
    },
    # Concepto distinto de 'marca': el umbral aquí es N<10, no 6 o menos.
    "marcapalu": {
        "1": "Estudiantes insuficientes para distribución por estándar (N<10)",
    },
    "noaplica": {
        "0": "Rindió Simce",
        "1": "Curso ausente sin material aplicado",
        "2": "Curso de estudiantes integrados (NEEP)",
        "3": "Curso ausente con algún cuestionario aplicado",
        "4": "Estudiantes renuncian a responder",
        "5": "Establecimiento o curso en paro",
        "6": "Pérdida total de material",
        "7": "Comunidad educativa se niega a rendir",
        # Código presente en los datos de 2024 (1 establecimiento) pero ausente
        # de todos los glosarios oficiales. Se etiqueta como no documentado en
        # lugar de inventarle un significado.
        "8": "Sin documentar en glosario (código 8)",
    },
}

#: Reasignación de códigos por año, cuando el esquema del año NO coincide con
#: el esquema canónico. Formato: {(concepto, año): {código_del_año: código_canónico}}.
#: Único caso real detectado: en 2023-2025 los códigos de ``marcadif`` cambian
#: (1 pasa a significar "comparación no válida" y 2 "estudiantes insuficientes").
REASIGNACION_POR_ANIO: dict[tuple[str, int], dict[str, str]] = {
    ("marcadif", anio): {"1": "2", "2": "1"} for anio in (2023, 2024, 2025)
}


def codigo_canonico(valor: object, concepto: str, anio: int | None = None) -> str | None:
    """Resuelve un valor crudo (código o texto) a su código canónico.

    Parameters
    ----------
    valor : object
        Valor tal como viene en el CSV: código (``1``, ``"1.0"``) o texto.
    concepto : str
        Concepto del catálogo (``"dependencia1"``, ``"marcadif"``, ...).
    anio : int | None, optional
        Año del dato. Necesario cuando el esquema de códigos del año difiere
        del canónico (ver :data:`REASIGNACION_POR_ANIO`).

    Returns
    -------
    str | None
        Código canónico, o ``None`` si no se pudo resolver.

    Examples
    --------
    >>> codigo_canonico("Corporación Municipal", "dependencia1")
    '1'
    >>> codigo_canonico("1", "marcadif", 2022)   # esquema antiguo
    '1'
    >>> codigo_canonico("1", "marcadif", 2023)   # códigos permutados
    '2'
    """
    if valor is None:
        return None
    bruto = str(valor).strip()
    if bruto == "" or bruto.lower() in {"nan", "none", "<na>"}:
        return None

    # ¿Es un código?
    cod = None
    try:
        f = float(bruto.replace(",", "."))
        cod = str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        cod = None

    if cod is None:  # es texto: se busca en la tabla de alias explícita
        return _buscar_alias(bruto, concepto)

    # Es código: se aplica la reasignación del año si corresponde.
    if anio is not None:
        mapa = REASIGNACION_POR_ANIO.get((concepto, int(anio)))
        if mapa:
            cod = mapa.get(cod, cod)
    return cod


def _buscar_alias(texto: str, concepto: str) -> str | None:
    """Busca un texto en los alias del concepto, comparando normalizado."""
    objetivo = _norm_alias(texto)
    for alias, cod in ALIAS_TEXTO.get(concepto, {}).items():
        if _norm_alias(alias) == objetivo:
            return cod
    return None


def _norm_alias(s: str) -> str:
    """Normaliza para comparar alias: minúsculas, sin acentos ni puntuación."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def categoria_canonica(valor: object, concepto: str,
                       anio: int | None = None) -> str | None:
    """Resuelve un valor crudo a su **categoría armonizada** de la serie.

    Parameters
    ----------
    valor : object
    concepto : str
    anio : int | None, optional

    Returns
    -------
    str | None
        Categoría canónica, o ``None`` si no se pudo resolver.

    Examples
    --------
    >>> categoria_canonica("Evaluación actual no permite comparar", "marcadif", 2015)
    'Comparación no válida'
    >>> categoria_canonica("1", "marcadif", 2023)
    'Comparación no válida'
    """
    cod = codigo_canonico(valor, concepto, anio)
    if cod is None:
        return None
    return CATEGORIAS.get(concepto, {}).get(cod)
