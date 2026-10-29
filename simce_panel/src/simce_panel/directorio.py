"""
directorio.py
=============

Cruce del panel RBD con el Directorio Oficial de Establecimientos para
georreferenciar los colegios (latitud/longitud) y, opcionalmente, anexar
variables administrativas (dependencia, ruralidad, región, comuna).

El cruce es *por año*: para cada año del panel SIMCE se usa el Directorio del
mismo año, de modo que un establecimiento que cambió de coordenadas o que solo
existe en ciertos años se empareja correctamente. El join es siempre LEFT
sobre el panel SIMCE, por lo que ninguna fila SIMCE se pierde; si un RBD no
tiene coordenadas en el Directorio de ese año, los valores quedan como NaN.

Funciones principales
---------------------
- :func:`mapear_directorios`     : {anio: ruta_csv} de Directorios disponibles.
- :func:`cargar_directorio_anio` : carga un Directorio y estandariza columnas.
- :func:`construir_tabla_geo`    : tabla (año, rbd, lat, lon, ...) de todos los años.
- :func:`enriquecer_rbd`         : LEFT JOIN del panel RBD con la tabla geo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config
from .columnas import resolver_columna, resolver_columna_req
from .io_datos import a_numerico, extraer_anio, leer_csv

log = logging.getLogger(__name__)


def mapear_directorios(cfg: config.Config) -> dict[int, Path]:
    """Construye el mapa ``{anio: ruta}`` de los Directorios disponibles.

    Busca en la carpeta de Directorio todos los CSV y extrae el año de su
    nombre. Si hubiera más de un archivo para el mismo año, conserva el primero.

    Parameters
    ----------
    cfg : config.Config

    Returns
    -------
    dict[int, Path]
    """
    mapa: dict[int, Path] = {}
    for archivo in sorted(cfg.directorio_dir.glob("*.csv")):
        anio = extraer_anio(archivo.name)
        if anio is not None and anio not in mapa:
            mapa[anio] = archivo
    return mapa


def cargar_directorio_anio(
    ruta: str | Path,
    anio: int,
    *,
    cfg: config.Config,
    columnas_extra: list[str] | None = None,
    sanear_geo: bool = True,
) -> pd.DataFrame:
    """Carga un Directorio y devuelve una tabla geo estandarizada para un año.

    La salida tiene SIEMPRE estas columnas:
    ``año``, ``rbd``, ``latitud``, ``longitud`` (más las ``columnas_extra``
    administrativas que se soliciten y existan en el archivo).

    Las coordenadas se convierten a numérico tolerando coma decimal. El RBD se
    normaliza a entero (``Int64`` nullable) para que el join con el panel SIMCE
    sea exacto y no falle por diferencias de tipo (texto vs número).

    Parameters
    ----------
    ruta : str | Path
        Ruta del CSV del Directorio.
    anio : int
        Año al que corresponde el Directorio.
    cfg : config.Config
    columnas_extra : list[str] | None, optional
        Conceptos administrativos adicionales a extraer. Cada elemento debe ser
        una clave de :data:`config.ALIAS_COLUMNAS` (p. ej. ``"dependencia"``,
        ``"ruralidad"``, ``"region"``, ``"comuna"``). Se omiten silenciosamente
        los que no existan en el archivo de ese año.

    Returns
    -------
    pandas.DataFrame
    """
    df = leer_csv(ruta, sep=cfg.sep, encoding=cfg.encoding)

    col_rbd = resolver_columna_req(
        df, config.ALIAS_COLUMNAS["rbd"],
        concepto="rbd", contexto=f"Directorio {anio}",
    )
    col_lat = resolver_columna(df, config.ALIAS_COLUMNAS["latitud"])
    col_lon = resolver_columna(df, config.ALIAS_COLUMNAS["longitud"])

    if col_lat is None or col_lon is None:
        log.warning(
            "Directorio %s: no se hallaron coordenadas (lat=%s, lon=%s). "
            "Se rellenará con NaN.", anio, col_lat, col_lon,
        )

    salida = pd.DataFrame()
    salida[config.COL_ANIO] = pd.Series([anio] * len(df), dtype="int64")
    salida[config.COL_RBD] = _rbd_a_entero(df[col_rbd])
    salida[config.COL_LAT] = (
        a_numerico(df[col_lat]) if col_lat else pd.Series([pd.NA] * len(df))
    )
    salida[config.COL_LON] = (
        a_numerico(df[col_lon]) if col_lon else pd.Series([pd.NA] * len(df))
    )

    for concepto in (columnas_extra or []):
        alias = config.ALIAS_COLUMNAS.get(concepto)
        if not alias:
            log.warning("Concepto desconocido en columnas_extra: '%s'.", concepto)
            continue
        col = resolver_columna(df, alias)
        if col is not None:
            salida[concepto] = df[col].values
        else:
            log.info("Directorio %s: sin columna para '%s' (se omite).",
                     anio, concepto)

    # Validación y reparación de coordenadas (coma decimal, separadores de
    # miles, enteros sin punto decimal y lat/lon invertidas). Sin esto, un
    # valor como "-33456789" entraría al panel como una coordenada absurda.
    if sanear_geo and col_lat and col_lon:
        salida, rep = sanear_coordenadas(salida)
        if not rep.empty:
            malas = int(rep.loc[0, ["reparadas_formato", "invertidas_corregidas",
                                    "irreparables"]].sum())
            if malas:
                log.warning("Directorio %s: %d coordenadas requirieron atención "
                            "(%d reparadas, %d invertidas, %d irreparables).",
                            anio, malas, rep.loc[0, "reparadas_formato"],
                            rep.loc[0, "invertidas_corregidas"],
                            rep.loc[0, "irreparables"])

    # Un RBD debería ser único por año en el Directorio; si hay duplicados,
    # se conserva el primero para no multiplicar filas en el join.
    antes = len(salida)
    salida = salida.drop_duplicates(subset=[config.COL_RBD], keep="first")
    if len(salida) < antes:
        log.info("Directorio %s: %d RBD duplicados eliminados.",
                 anio, antes - len(salida))
    return salida


def _rbd_a_entero(serie: pd.Series) -> pd.Series:
    """Convierte la columna RBD a entero nullable (``Int64``).

    Tolera valores con decimales espurios (``"1234.0"``) o texto. Los no
    convertibles quedan como ``<NA>``.
    """
    s = pd.to_numeric(
        serie.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return s.round().astype("Int64")


def construir_tabla_geo(
    cfg: config.Config,
    anios: list[int],
    *,
    columnas_extra: list[str] | None = None,
) -> pd.DataFrame:
    """Construye la tabla geo (año, rbd, lat, lon, ...) para los años indicados.

    Parameters
    ----------
    cfg : config.Config
    anios : list[int]
        Años a incluir (normalmente los presentes en el panel RBD).
    columnas_extra : list[str] | None, optional
        Conceptos administrativos adicionales (ver :func:`cargar_directorio_anio`).

    Returns
    -------
    pandas.DataFrame
        Tabla apilada con una fila por (año, rbd).
    """
    mapa = mapear_directorios(cfg)
    bloques: list[pd.DataFrame] = []
    for anio in sorted(set(anios)):
        ruta = mapa.get(anio)
        if ruta is None:
            log.warning("No hay Directorio para el año %s; ese año quedará "
                        "sin coordenadas.", anio)
            continue
        geo = cargar_directorio_anio(
            ruta, anio, cfg=cfg, columnas_extra=columnas_extra
        )
        log.info("Directorio %s: %d RBD con geo.", anio, len(geo))
        bloques.append(geo)
    if not bloques:
        return pd.DataFrame(
            columns=[config.COL_ANIO, config.COL_RBD,
                     config.COL_LAT, config.COL_LON]
        )
    return pd.concat(bloques, ignore_index=True)


def enriquecer_rbd(
    panel_rbd: pd.DataFrame,
    cfg: config.Config,
    *,
    columnas_extra: list[str] | None = None,
    guardar: bool = True,
) -> pd.DataFrame:
    """Anexa coordenadas (y columnas administrativas) al panel RBD por LEFT JOIN.

    Conserva TODAS las columnas y filas del panel SIMCE RBD. El emparejamiento
    se hace por ``(año, rbd)`` para respetar la dimensión temporal.

    Parameters
    ----------
    panel_rbd : pandas.DataFrame
        Panel RBD unificado (salida de :func:`unificacion.unificar_nivel`).
    cfg : config.Config
    columnas_extra : list[str] | None, optional
        Conceptos administrativos adicionales a anexar desde el Directorio.
    guardar : bool, optional
        Si ``True`` (por defecto) sobrescribe el CSV RBD unificado con la
        versión enriquecida.

    Returns
    -------
    pandas.DataFrame
        Panel RBD con ``latitud``/``longitud`` (y extras) anexadas.
    """
    from .io_datos import escribir_csv

    if panel_rbd.empty:
        log.warning("El panel RBD está vacío; no se enriquece.")
        return panel_rbd

    panel = panel_rbd.copy()
    col_rbd_panel = resolver_columna_req(
        panel, config.ALIAS_COLUMNAS["rbd"],
        concepto="rbd", contexto="panel RBD unificado",
    )
    # Llave de cruce normalizada en una columna temporal para no alterar la
    # columna rbd original del SIMCE.
    panel["_rbd_key"] = _rbd_a_entero(panel[col_rbd_panel])

    anios = panel[config.COL_ANIO].dropna().astype(int).unique().tolist()
    geo = construir_tabla_geo(cfg, anios, columnas_extra=columnas_extra)

    if geo.empty:
        log.warning("Sin tabla geo; se devuelven coordenadas NaN.")
        panel[config.COL_LAT] = pd.NA
        panel[config.COL_LON] = pd.NA
    else:
        geo = geo.rename(columns={config.COL_RBD: "_rbd_key"})
        panel = panel.merge(
            geo, on=[config.COL_ANIO, "_rbd_key"], how="left",
        )

    panel = panel.drop(columns=["_rbd_key"])

    # Reporte de cobertura de georreferenciación.
    if config.COL_LAT in panel.columns:
        cobertura = panel[config.COL_LAT].notna().mean() * 100
        log.info("Cobertura de coordenadas: %.1f%% de las filas RBD.", cobertura)

    if guardar:
        ruta = cfg.ruta_salida(config.NOMBRES_SALIDA["rbd"])
        escribir_csv(panel, ruta, sep=cfg.sep, encoding=cfg.encoding)
        log.info("Guardado RBD enriquecido: %s", ruta)

    return panel


#: Conceptos territoriales que conviene tomar del Directorio y NO del archivo
#: SIMCE. Motivo verificado sobre los datos reales: en el SIMCE 2015 las
#: columnas ``cod_com_rbd``, ``cod_pro_rbd`` y ``cod_deprov_rbd`` contienen
#: todas el mismo valor (un índice de DEPROV, 54 valores distintos) en lugar
#: del código territorial oficial (~330 comunas), y en 2018 los nombres de
#: comuna vienen truncados ("Alto Biob", "Alto Del").
CONCEPTOS_TERRITORIALES: tuple[str, ...] = ("region", "comuna")


def anexar_territorio(
    panel_rbd: pd.DataFrame,
    cfg: config.Config,
    *,
    conceptos: tuple[str, ...] = CONCEPTOS_TERRITORIALES,
    sufijo: str = "_dir",
) -> pd.DataFrame:
    """Anexa las variables territoriales tomadas del Directorio Oficial.

    Las columnas territoriales del propio archivo SIMCE no son fiables en toda
    la serie (ver :data:`CONCEPTOS_TERRITORIALES`). Esta función trae región y
    comuna desde el Directorio del año correspondiente, en columnas nuevas con
    sufijo ``_dir``, sin tocar las originales.

    Parameters
    ----------
    panel_rbd : pandas.DataFrame
        Panel RBD largo (con columnas ``año`` y ``rbd``).
    cfg : config.Config
    conceptos : tuple[str, ...], optional
        Claves de :data:`config.ALIAS_COLUMNAS` a traer.
    sufijo : str, optional
        Sufijo de las columnas creadas (por defecto ``_dir``).

    Returns
    -------
    pandas.DataFrame
        El panel con las columnas ``<concepto><sufijo>`` anexadas.
    """
    from .columnas import resolver_columna_req

    panel = panel_rbd.copy()
    col_rbd = resolver_columna_req(panel, config.ALIAS_COLUMNAS["rbd"],
                                   concepto="rbd", contexto="panel RBD")
    panel["_rbd_key"] = _rbd_a_entero(panel[col_rbd])
    anios = panel[config.COL_ANIO].dropna().astype(int).unique().tolist()

    geo = construir_tabla_geo(cfg, anios, columnas_extra=list(conceptos))
    disponibles = [c for c in conceptos if c in geo.columns]
    if not disponibles:
        log.warning("El Directorio no aportó ninguna variable territorial.")
        return panel.drop(columns=["_rbd_key"])

    geo = (geo[[config.COL_ANIO, config.COL_RBD] + disponibles]
           .rename(columns={config.COL_RBD: "_rbd_key",
                            **{c: f"{c}{sufijo}" for c in disponibles}}))
    panel = panel.merge(geo, on=[config.COL_ANIO, "_rbd_key"], how="left")
    panel = panel.drop(columns=["_rbd_key"])

    for c in disponibles:
        cob = panel[f"{c}{sufijo}"].notna().mean() * 100
        log.info("Territorio '%s' desde Directorio: %.1f%% de cobertura.", c, cob)
    return panel


#: Zonas geográficas válidas del territorio nacional, como pares de rangos
#: ``(lat_min, lat_max, lon_min, lon_max)``. Se validan como ZONAS y no como
#: dos rangos independientes: así se admite la Antártica chilena sin perder la
#: capacidad de detectar latitud y longitud intercambiadas.
#:
#: La zona antártica es imprescindible: el Directorio incluye la Escuela F-50
#: de Villa Las Estrellas (Isla Rey Jorge, Base Frei), en funcionamiento hasta
#: 2018, con coordenadas cercanas a -62,20 / -58,96.
ZONAS_CHILE: dict[str, tuple[float, float, float, float]] = {
    "continental_insular": (-56.0, -17.0, -110.0, -66.0),  # incl. Isla de Pascua
    "antartica": (-90.0, -60.0, -90.0, -53.0),             # Territorio Chileno Antártico
}

#: Rangos amplios derivados de las zonas (solo para diagnósticos por columna).
LIMITES_CHILE: dict[str, tuple[float, float]] = {
    "latitud": (-90.0, -17.0),
    "longitud": (-110.0, -53.0),
}


def _en_alguna_zona(lat: float, lon: float,
                    zonas: dict[str, tuple[float, float, float, float]] | None = None
                    ) -> str | None:
    """Devuelve el nombre de la zona que contiene el punto, o ``None``."""
    for nombre, (la, lb, oa, ob) in (zonas or ZONAS_CHILE).items():
        if la <= lat <= lb and oa <= lon <= ob:
            return nombre
    return None


def _reparar_coordenada(valor: object, lo: float, hi: float) -> tuple[float | None, str]:
    """Intenta interpretar un valor como coordenada dentro del rango ``[lo, hi]``.

    Estrategias, en orden:

    1. Conversión directa (tolera coma decimal).
    2. Eliminación de separadores de miles (``"-3.345.678"``).
    3. Reescalado de enteros sin punto decimal (``"-33456789"`` -> ``-33.456789``),
       dividiendo por potencias de 10 hasta caer en el rango válido.

    Returns
    -------
    tuple[float | None, str]
        ``(valor, estrategia)``; estrategia es ``"ok"``, ``"miles"``,
        ``"reescalado"``, ``"vacio"`` o ``"irreparable"``.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None, "vacio"
    bruto = str(valor).strip().replace(" ", "")
    if bruto == "" or bruto.lower() in {"nan", "none", "<na>"}:
        return None, "vacio"

    def _en_rango(x: float) -> bool:
        return lo <= x <= hi

    # 1) Directo (coma decimal -> punto).
    try:
        v = float(bruto.replace(",", "."))
        if _en_rango(v):
            return v, "ok"
    except ValueError:
        v = None

    # 2) Separadores de miles: varios puntos, o punto con grupos de 3 dígitos.
    sin_miles = bruto
    if bruto.count(".") > 1:
        sin_miles = bruto.replace(".", "")
    elif bruto.count(",") > 1:
        sin_miles = bruto.replace(",", "")
    try:
        v2 = float(sin_miles.replace(",", "."))
        if _en_rango(v2):
            return v2, "miles"
    except ValueError:
        v2 = None

    # 3) Entero sin separador decimal: se reescala dividiendo por 10^k.
    for base in (v, v2):
        if base is None:
            continue
        x = base
        for _ in range(10):
            if _en_rango(x):
                return x, "reescalado" if x != base else "ok"
            if abs(x) < abs(lo):     # ya es demasiado pequeño
                break
            x = x / 10.0
    return None, "irreparable"


def _valores_candidatos(valor: object) -> list[float]:
    """Devuelve las interpretaciones numéricas posibles de un valor crudo.

    Incluye la conversión directa (con coma decimal), la eliminación de
    separadores de miles y los reescalados por potencias de 10 (para enteros
    guardados sin punto decimal). El cero se descarta: en los Directorios de
    2015 y 2016 se usa como marca de "sin coordenada", no como una ubicación.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return []
    bruto = str(valor).strip().replace(" ", "")
    if bruto == "" or bruto.lower() in {"nan", "none", "<na>"}:
        return []

    crudos: list[float] = []
    for texto in {bruto,
                  bruto.replace(".", "") if bruto.count(".") > 1 else bruto,
                  bruto.replace(",", "") if bruto.count(",") > 1 else bruto}:
        try:
            crudos.append(float(texto.replace(",", ".")))
        except ValueError:
            continue

    candidatos: list[float] = []
    for x in crudos:
        if x == 0:            # 0 = ausencia de dato, no una coordenada
            continue
        v = x
        for _ in range(10):
            if v not in candidatos:
                candidatos.append(v)
            if abs(v) < 1:
                break
            v = v / 10.0
    return candidatos


def _resolver_par(vlat: object, vlon: object,
                  corregir_invertidas: bool = True
                  ) -> tuple[float | None, float | None, str]:
    """Resuelve un par (latitud, longitud) validándolo contra las zonas de Chile.

    Evalúa el par completo —no cada valor por separado— porque la validez
    geográfica depende de la combinación: -62,20 solo es una latitud válida si
    la longitud acompaña dentro del Territorio Chileno Antártico.

    Returns
    -------
    tuple[float | None, float | None, str]
        ``(lat, lon, estado)`` con estado ``"ok"``, ``"reparado"``,
        ``"invertida"``, ``"vacio"``, ``"incompleta"`` o ``"irreparable"``.
    """
    ca = _valores_candidatos(vlat)
    cb = _valores_candidatos(vlon)

    if not ca and not cb:
        return None, None, "vacio"
    if not ca or not cb:
        return None, None, "incompleta"

    # Orientación normal: se prefiere la interpretación directa (primera).
    for i, a in enumerate(ca):
        for j, b in enumerate(cb):
            if _en_alguna_zona(a, b):
                return a, b, ("ok" if i == 0 and j == 0 else "reparado")

    # Orientación invertida.
    if corregir_invertidas:
        for a in cb:
            for b in ca:
                if _en_alguna_zona(a, b):
                    return a, b, "invertida"

    return None, None, "irreparable"


def sanear_coordenadas(
    df: pd.DataFrame,
    *,
    col_lat: str = config.COL_LAT,
    col_lon: str = config.COL_LON,
    limites: dict[str, tuple[float, float]] | None = None,
    corregir_invertidas: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida y repara las coordenadas, dejando en NaN las irrecuperables.

    Problemas que resuelve, todos observados en bases públicas chilenas:

    - Coma decimal (``"-33,4567"``) — ya cubierto por :func:`io_datos.a_numerico`.
    - Separadores de miles (``"-3.345.678"``), que producían ``NaN``.
    - Enteros sin punto decimal (``"-33456789"`` por ``-33.456789``), que
      producían valores absurdos como ``-33456789`` sin avisar.
    - Latitud y longitud **intercambiadas** en algunas filas.

    Cualquier valor que no logre caer dentro de los límites de Chile queda como
    ``NaN``: es preferible perder el punto a ubicar un colegio en otro
    continente y contaminar el análisis espacial.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con columnas de latitud y longitud.
    col_lat, col_lon : str, optional
    limites : dict[str, tuple[float, float]] | None, optional
        Rangos válidos. Por defecto :data:`LIMITES_CHILE`.
    corregir_invertidas : bool, optional
        Si ``True``, cuando lat cae en el rango de lon y viceversa, se
        intercambian.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(df_corregido, reporte)``. El reporte cuenta cuántos valores se
        arreglaron con cada estrategia y cuántos quedaron sin coordenada.
    """
    lim = limites or LIMITES_CHILE
    out = df.copy()
    la, lb = lim["latitud"]
    oa, ob = lim["longitud"]

    if col_lat not in out.columns or col_lon not in out.columns:
        log.warning("Faltan columnas de coordenadas (%s / %s).", col_lat, col_lon)
        return out, pd.DataFrame()

    lats, lons, estados = [], [], []
    for vlat, vlon in zip(out[col_lat], out[col_lon]):
        a, b, estado = _resolver_par(vlat, vlon, corregir_invertidas)
        lats.append(a)
        lons.append(b)
        estados.append(estado)

    out[col_lat] = lats
    out[col_lon] = lons
    est = pd.Series(estados)

    reporte = pd.DataFrame([{
        "n_filas": len(est),
        "ok": int((est == "ok").sum()),
        "reparadas_formato": int((est == "reparado").sum()),
        "invertidas_corregidas": int((est == "invertida").sum()),
        "sin_coordenada": int((est == "vacio").sum()),
        "incompletas": int((est == "incompleta").sum()),
        "irreparables": int((est == "irreparable").sum()),
    }])

    if reporte.loc[0, "reparadas_formato"]:
        log.warning("Se repararon %d coordenadas mal formateadas.",
                    reporte.loc[0, "reparadas_formato"])
    if reporte.loc[0, "invertidas_corregidas"]:
        log.warning("%d filas tenían latitud y longitud invertidas; se corrigieron.",
                    reporte.loc[0, "invertidas_corregidas"])
    if reporte.loc[0, "incompletas"]:
        log.info("%d filas traían solo una de las dos coordenadas.",
                 reporte.loc[0, "incompletas"])
    if reporte.loc[0, "irreparables"]:
        log.warning("%d coordenadas con valores fuera del rango de Chile "
                    "quedaron como NaN (¿otro sistema de coordenadas?).",
                    reporte.loc[0, "irreparables"])
    return out, reporte


def inspeccionar_coordenadas_crudas(
    cfg: config.Config,
    anio: int,
    n: int = 10,
) -> pd.DataFrame:
    """Muestra los valores CRUDOS de coordenadas que no se pudieron interpretar.

    Sirve para diagnosticar un Directorio con muchas coordenadas irreparables:
    permite ver si vienen vacías, en otro sistema (UTM) o con un formato no
    contemplado.

    Parameters
    ----------
    cfg : config.Config
    anio : int
        Año del Directorio a inspeccionar.
    n : int, optional
        Número de ejemplos a devolver.

    Returns
    -------
    pandas.DataFrame
        Filas con las coordenadas crudas que no se resolvieron.
    """
    from .columnas import resolver_columna

    ruta = mapear_directorios(cfg).get(anio)
    if ruta is None:
        log.warning("No hay Directorio para %s.", anio)
        return pd.DataFrame()

    df = leer_csv(ruta, sep=cfg.sep, encoding=cfg.encoding)
    c_lat = resolver_columna(df, config.ALIAS_COLUMNAS["latitud"])
    c_lon = resolver_columna(df, config.ALIAS_COLUMNAS["longitud"])
    c_rbd = resolver_columna(df, config.ALIAS_COLUMNAS["rbd"])
    if not (c_lat and c_lon):
        log.warning("Directorio %s no tiene columnas de coordenadas.", anio)
        return pd.DataFrame()

    la, lb = LIMITES_CHILE["latitud"]
    oa, ob = LIMITES_CHILE["longitud"]
    problemas = []
    for _, r in df.iterrows():
        a, ea = _reparar_coordenada(r[c_lat], la, lb)
        b, eb = _reparar_coordenada(r[c_lon], oa, ob)
        if a is not None and b is not None:
            continue
        problemas.append({
            "rbd": r[c_rbd] if c_rbd else None,
            "lat_cruda": repr(r[c_lat]),
            "lon_cruda": repr(r[c_lon]),
            "estado_lat": ea, "estado_lon": eb,
        })
        if len(problemas) >= n:
            break
    return pd.DataFrame(problemas)


#: Formato canónico de coordenadas en los Directorios oficiales desde 2017:
#: grados decimales, signo negativo, coma como separador decimal y ~8 decimales
#: (p. ej. ``-18,48727459`` / ``-70,29473809``). El pipeline lo convierte a
#: punto decimal (``-18.48727459``) para que sea legible por software, y lo
#: escribe así en todas las salidas, preservando la precisión original.
FORMATO_CANONICO_COORDENADAS: str = "grados decimales, coma decimal, ~8 decimales"


def diagnostico_formato_coordenadas(
    cfg: config.Config,
    anios: list[int] | None = None,
    n_ejemplos: int = 3,
) -> pd.DataFrame:
    """Reporta qué formato de coordenadas usa el Directorio de cada año.

    Permite comparar todos los años contra el formato canónico de 2017 en
    adelante y localizar exactamente dónde y por qué falla la conversión.

    Parameters
    ----------
    cfg : config.Config
    anios : list[int] | None, optional
        Años a revisar. Si es ``None``, todos los Directorios disponibles.
    n_ejemplos : int, optional
        Cuántos valores crudos de muestra incluir por año.

    Returns
    -------
    pandas.DataFrame
        Por año: nombres de las columnas detectadas, cuántos valores usan coma
        o punto decimal, cuántos están vacíos, cuántos quedan fuera del rango
        de Chile, y ejemplos crudos de los problemáticos.
    """
    from .columnas import resolver_columna

    mapa = mapear_directorios(cfg)
    objetivo = sorted(anios) if anios else sorted(mapa)
    la, lb = LIMITES_CHILE["latitud"]
    oa, ob = LIMITES_CHILE["longitud"]

    filas = []
    for anio in objetivo:
        ruta = mapa.get(anio)
        if ruta is None:
            continue
        df = leer_csv(ruta, sep=cfg.sep, encoding=cfg.encoding, dtype=str)
        c_lat = resolver_columna(df, config.ALIAS_COLUMNAS["latitud"])
        c_lon = resolver_columna(df, config.ALIAS_COLUMNAS["longitud"])
        if not (c_lat and c_lon):
            filas.append({"año": anio, "col_lat": c_lat, "col_lon": c_lon,
                          "n_filas": len(df), "sin_columnas_geo": True})
            continue

        s = df[c_lat].astype(str).str.strip()
        vacios = int((s.isin(["", "nan", "None", "NaN"]) | df[c_lat].isna()).sum())
        con_coma = int(s.str.contains(",", na=False).sum())
        con_punto = int(s.str.contains(r"\.", na=False, regex=True).sum())
        sin_sep = len(s) - vacios - con_coma - con_punto

        malos = []
        n_fuera = 0
        for vlat, vlon in zip(df[c_lat], df[c_lon]):
            a, ea = _reparar_coordenada(vlat, la, lb)
            b, eb = _reparar_coordenada(vlon, oa, ob)
            if a is None or b is None:
                if not (ea == "vacio" and eb == "vacio"):
                    n_fuera += 1
                    if len(malos) < n_ejemplos:
                        malos.append(f"{vlat!r} / {vlon!r}")

        filas.append({
            "año": anio,
            "col_lat": c_lat, "col_lon": c_lon,
            "n_filas": len(df),
            "vacios": vacios,
            "con_coma_decimal": con_coma,
            "con_punto_decimal": con_punto,
            "sin_separador": sin_sep,
            "no_convertibles": n_fuera,
            "ejemplos_problematicos": malos,
        })
    return pd.DataFrame(filas)
