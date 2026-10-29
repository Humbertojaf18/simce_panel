"""
covariables.py
==============

Covariables adicionales pedidas en la revisión del 1-oct-2026 (comentarios 15
y 23): matrícula, tasa de participación, IVE de JUNAEB, pobreza comunal,
densidad de población comunal y distancia a la capital regional.

La distancia a la capital regional se calcula con los datos del propio panel.
Las demás requieren descargar archivos públicos; las funciones de carga
documentan las columnas esperadas y validan el resultado, pero los nombres
exactos de las columnas deben confirmarse al descargar cada fuente.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Comuna capital de cada región, por código único territorial (CUT).
#: Las capitales regionales tienen el código región * 1000 + 101.
CAPITAL_REGIONAL: dict[int, int] = {r: r * 1000 + 101 for r in range(1, 17)}


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - np.asarray(lon1)) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def distancia_capital_regional(df: pd.DataFrame, *, col_region: str = "region_dir",
                               col_comuna: str = "comuna_dir",
                               col_lat: str = "latitud",
                               col_lon: str = "longitud") -> pd.Series:
    """Distancia (km) de cada establecimiento a la capital de su región.

    La posición de la capital se aproxima por el centroide de los
    establecimientos de su comuna en el propio panel, de modo que no requiere
    datos externos. Mide la periferia administrativa, distinta de la
    centralidad nacional ya incluida en el dataset analítico.
    """
    d = df[[col_region, col_comuna, col_lat, col_lon]].copy()
    for c in (col_region, col_comuna):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    centro = d.groupby(col_comuna)[[col_lat, col_lon]].mean()
    cap = d[col_region].map(CAPITAL_REGIONAL)
    lat_c = cap.map(centro[col_lat]); lon_c = cap.map(centro[col_lon])
    sin = int(lat_c.isna().sum())
    if sin:
        log.warning("%d establecimientos sin capital regional ubicable.", sin)
    dist = pd.Series(_haversine_km(d[col_lat], d[col_lon], lat_c, lon_c),
                     index=df.index, name="dist_capital_regional_km")
    # Control de coherencia: fuera de Valparaíso (Rapa Nui, Juan Fernández) y
    # Magallanes (Antártica), ningún establecimiento debería estar a más de
    # 800 km de su capital regional. Un valor mayor delata una coordenada
    # errónea que pasó la validación por zonas del Directorio.
    sospecha = (dist > 800) & ~d[col_region].isin([5, 12])
    if sospecha.any():
        log.warning("%d establecimientos a más de 800 km de su capital regional "
                    "(coordenada probablemente errónea): RBD %s", int(sospecha.sum()),
                    df.loc[sospecha[sospecha].index].get("rbd", pd.Series()).tolist())
    dist.attrs["sospechosos"] = sospecha
    return dist


def tasa_participacion(ancho: pd.DataFrame, matricula: pd.DataFrame, *,
                       materia: str = "mate", col_rbd: str = "rbd",
                       anios=(2015, 2016, 2017, 2018, 2022, 2023, 2024, 2025)) -> pd.DataFrame:
    """Razón entre evaluados y matriculados de II medio, por establecimiento y año.

    Responde la alternativa (ii) del comentario 15: si en 2022 rindieron la
    prueba menos estudiantes en las escuelas vulnerables, y los ausentes eran
    los de menor desempeño, el promedio de esas escuelas queda inflado y su
    caída subestimada.

    Parameters
    ----------
    matricula : pandas.DataFrame
        Columnas ``rbd``, ``anio`` y ``matricula_2m`` (matrícula de II medio),
        construidas desde los datos abiertos de matrícula del MINEDUC.
    """
    ev = pd.concat([pd.DataFrame({col_rbd: ancho[col_rbd], "anio": t,
                                  "evaluados": pd.to_numeric(ancho[f"nalu_{materia}2m_rbd_{t}"],
                                                             errors="coerce")})
                    for t in anios])
    out = ev.merge(matricula, on=[col_rbd, "anio"], how="left")
    out["tasa_participacion"] = out["evaluados"] / out["matricula_2m"]
    fuera = int((out["tasa_participacion"] > 1.05).sum())
    if fuera:
        log.warning("%d registros con tasa > 1,05: revisar el nivel de la matrícula.", fuera)
    return out


def cargar_comunal(ruta: str, *, col_comuna: str, col_valor: str,
                   nombre: str, sep: str = ";") -> pd.Series:
    """Carga una covariable comunal (pobreza, densidad) indexada por código CUT.

    Fuentes previstas: pobreza comunal, estimaciones para áreas pequeñas del
    Ministerio de Desarrollo Social y Familia basadas en la CASEN; densidad,
    población y superficie comunal del INE.
    """
    from .io_datos import leer_csv
    t = leer_csv(ruta, sep=sep) if ruta.endswith(".csv") else pd.read_excel(ruta)
    s = pd.to_numeric(t[col_valor], errors="coerce")
    s.index = pd.to_numeric(t[col_comuna], errors="coerce").astype("Int64")
    s.name = nombre
    log.info("%s: %d comunas, %d sin valor.", nombre, s.index.nunique(), int(s.isna().sum()))
    return s


def cargar_ive(ruta: str, *, col_rbd: str = "rbd", col_ive: str = "ive",
               sep: str = ";") -> pd.Series:
    """Carga el Índice de Vulnerabilidad Escolar (JUNAEB) por establecimiento."""
    from .io_datos import leer_csv
    t = leer_csv(ruta, sep=sep) if ruta.endswith(".csv") else pd.read_excel(ruta)
    s = pd.to_numeric(t[col_ive], errors="coerce")
    # El IVE se publica en porcentaje o en proporción según el año.
    if s.max() > 1.5:
        s = s / 100.0
    s.index = pd.to_numeric(t[col_rbd], errors="coerce").astype("Int64")
    return s.rename("ive")


# --------------------------------------------------------------------------- #
# Procesadores de archivos oficiales (detección automática de columnas)
# --------------------------------------------------------------------------- #

def detectar_columna(columnas, *patrones, obligatoria: bool = True):
    """Primera columna cuyo nombre (sin mayúsculas ni tildes) contiene todos los patrones."""
    import unicodedata
    norm = lambda s: unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    for c in columnas:
        if all(norm(p) in norm(c) for p in patrones):
            return c
    if obligatoria:
        raise KeyError(f"No se encontró una columna con {patrones} entre: {list(columnas)[:25]}")
    return None


#: Códigos de enseñanza media para jóvenes (humanístico-científica y
#: técnico-profesional). VERIFICAR contra el libro de códigos del archivo
#: descargado: la validación de :func:`validar_matricula` delata un error.
COD_ENSE_MEDIA_JOVENES: tuple[int, ...] = (310, 410, 510, 610, 710, 810, 910)


def procesar_matricula_estudiantes(ruta: str, anio: int, *, grado: int = 2,
                                   cod_ense=COD_ENSE_MEDIA_JOVENES, sep: str | None = None,
                                   chunksize: int = 500_000) -> pd.DataFrame:
    """Cuenta la matrícula de un grado por establecimiento desde el archivo por estudiante.

    El archivo de matrícula por estudiante de datos abiertos del MINEDUC tiene
    millones de filas; se lee por partes y solo con las columnas necesarias
    (RBD, código de enseñanza y grado), detectadas automáticamente.

    Returns
    -------
    pandas.DataFrame
        Columnas ``rbd``, ``anio``, ``matricula_2m``.
    """
    if sep is None:
        with open(ruta, encoding="latin-1") as f:
            cab = f.readline()
        sep = ";" if cab.count(";") > cab.count(",") else ","
        cols = cab.strip().split(sep)
    else:
        with open(ruta, encoding="latin-1") as f:
            cols = f.readline().strip().split(sep)
    c_rbd = detectar_columna(cols, "rbd")
    c_ens = detectar_columna(cols, "cod_ense")
    c_gra = detectar_columna(cols, "cod_grado")
    conteo = []
    for parte in pd.read_csv(ruta, sep=sep, usecols=[c_rbd, c_ens, c_gra],
                             encoding="latin-1", chunksize=chunksize, low_memory=False):
        ens = pd.to_numeric(parte[c_ens], errors="coerce")
        gra = pd.to_numeric(parte[c_gra], errors="coerce")
        sel = parte[ens.isin(cod_ense) & (gra == grado)]
        conteo.append(sel.groupby(c_rbd).size())
    s = pd.concat(conteo).groupby(level=0).sum()
    out = s.rename("matricula_2m").reset_index().rename(columns={c_rbd: "rbd"})
    out["rbd"] = pd.to_numeric(out["rbd"], errors="coerce").astype("Int64")
    out["anio"] = anio
    log.info("Matrícula %d: %d establecimientos, %d estudiantes de %d.º medio.",
             anio, len(out), int(out["matricula_2m"].sum()), grado)
    return out[["rbd", "anio", "matricula_2m"]]


def validar_matricula(matricula: pd.DataFrame, ancho: pd.DataFrame, *,
                      materia: str = "mate") -> pd.DataFrame:
    """Compara la matrícula calculada con los evaluados del SIMCE.

    En años normales rinde la prueba la gran mayoría de los matriculados, de
    modo que la mediana de evaluados/matrícula debería rondar 0,85-0,95. Una
    mediana muy inferior indica que el filtro de nivel incluyó otros grados
    o modalidades; muy superior a 1, que excluyó alguna modalidad de jóvenes.
    """
    part = tasa_participacion(ancho, matricula, materia=materia)
    tabla = (part.dropna(subset=["tasa_participacion"]).groupby("anio")["tasa_participacion"]
             .describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].round(3))
    for anio, med in tabla["50%"].items():
        if not 0.7 <= med <= 1.05:
            log.warning("Año %s: mediana evaluados/matrícula = %.2f, fuera de lo "
                        "esperable. Revisa los códigos de enseñanza y grado.", anio, med)
    return tabla


def procesar_tabla_externa(ruta: str, *, clave: tuple[str, ...], valor: tuple[str, ...],
                           nombre: str, escala_porcentaje: bool = False, hoja=0) -> pd.Series:
    """Lee una tabla oficial (CSV o Excel) detectando la columna clave y la de valor.

    Sirve para el IVE de JUNAEB (clave ``("rbd",)``), la pobreza comunal del
    Ministerio de Desarrollo Social (clave ``("codigo",)`` o ``("comuna",)``) y la
    población o densidad del INE. Si la tabla trae filas de encabezado antes de
    los nombres de columna, se busca automáticamente la primera fila que
    contiene la clave.
    """
    if ruta.lower().endswith((".xlsx", ".xls")):
        crudo = pd.read_excel(ruta, sheet_name=hoja, header=None)
    else:
        from .io_datos import leer_csv
        crudo = leer_csv(ruta, header=None)
    fila = next(i for i in range(min(30, len(crudo)))
                if any(all(p.lower() in str(v).lower() for p in clave) for v in crudo.iloc[i]))
    t = crudo.iloc[fila + 1:].copy(); t.columns = crudo.iloc[fila].astype(str)
    c_k = detectar_columna(t.columns, *clave); c_v = detectar_columna(t.columns, *valor)
    s = pd.to_numeric(t[c_v].astype(str).str.replace(",", ".").str.replace("%", ""), errors="coerce")
    if escala_porcentaje and s.max() > 1.5:
        s = s / 100.0
    s.index = pd.to_numeric(t[c_k], errors="coerce").astype("Int64")
    s = s[s.index.notna()].groupby(level=0).mean()
    log.info("%s: %d claves leídas desde '%s' / '%s'.", nombre, len(s), c_k, c_v)
    return s.rename(nombre)


# --------------------------------------------------------------------------- #
# Índices de vulnerabilidad de JUNAEB (IVE e IVM)
# --------------------------------------------------------------------------- #
#
# Dos productos distintos, publicados en https://www.junaeb.cl/indices-de-vulnerabilidad/
#
# - IVE (Índice de Vulnerabilidad Escolar): porcentaje de estudiantes del grupo
#   prioritario por establecimiento, básica y media juntas. Trae además la
#   matrícula total del establecimiento.
# - IVM (Índice de Vulnerabilidad Multidimensional): puntaje por establecimiento
#   y nivel; su hoja "Media" es específica de enseñanza media.
#
# Ambos se fijan al período previo a la pandemia, igual que el GSE, para que la
# covariable describa la composición anterior al fenómeno y no incorpore
# cambios que podrían ser consecuencia de él.

import re as _re


def _norm(s) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def _hoja_con_rbd(ruta: str, preferir: tuple[str, ...] = ()):
    """Primera hoja con encabezado RBD, priorizando las que contienen ``preferir``."""
    x = pd.ExcelFile(ruta)
    hojas = sorted(x.sheet_names, key=lambda h: 0 if any(p in _norm(h) for p in preferir) else 1)
    for h in hojas:
        crudo = pd.read_excel(x, sheet_name=h, header=None, nrows=40)
        for i in range(len(crudo)):
            if any(_re.fullmatch(r"(id_)?rbd", _norm(v).strip()) for v in crudo.iloc[i]):
                t = pd.read_excel(x, sheet_name=h, header=i)
                t.columns = [str(c).strip() for c in t.columns]
                return h, t
    raise ValueError(f"Ninguna hoja de {ruta} tiene una columna RBD.")


def leer_ive(ruta: str, anio: int) -> pd.DataFrame:
    """IVE por establecimiento: proporción de estudiantes prioritarios y matrícula.

    La columna del índice se busca con la palabra completa "IVE", para no
    confundirla con columnas como "NIVEL". Si el archivo separa básica y media,
    se prefiere la de media, que corresponde a 2.º medio.
    """
    # Algunos años traen una sola hoja (básica y media juntas) y otros separan
    # "BÁSICA", "MEDIA" y "COMUNA". Para 2.º medio se prefiere la de media.
    hoja, t = _hoja_con_rbd(ruta, preferir=("media", "establec"))
    if "basica" in _norm(hoja) and "media" not in _norm(hoja):
        log.warning("IVE %d: se usó la hoja '%s' (básica); no se encontró una de media.", anio, hoja)
    cand = [c for c in t.columns if _re.search(r"\bive\b", _norm(c))]
    if not cand:
        raise KeyError(f"{ruta}: sin columna IVE en la hoja '{hoja}'.")
    col = (next((c for c in cand if "media" in _norm(c)), None)
           or next((c for c in cand if "prioritario" in _norm(c)), None) or cand[0])
    c_rbd = next(c for c in t.columns if _re.fullmatch(r"(id_)?rbd", _norm(c).strip()))
    c_mat = next((c for c in t.columns if "matricula" in _norm(c)), None)
    out = pd.DataFrame({"rbd": pd.to_numeric(t[c_rbd], errors="coerce"),
                        "ive": pd.to_numeric(t[col], errors="coerce")})
    if c_mat:
        out["matricula_total"] = pd.to_numeric(t[c_mat], errors="coerce")
    if out["ive"].max() > 1.5:          # publicado en porcentaje
        out["ive"] = out["ive"] / 100.0
    out = out.dropna(subset=["rbd", "ive"]).assign(anio=anio)
    out["rbd"] = out["rbd"].astype(int)
    log.info("IVE %d: hoja '%s', columna '%s', %d establecimientos.", anio, hoja, col, len(out))
    out.attrs["columna"] = col
    out.attrs["hoja"] = hoja
    return out.drop_duplicates("rbd")


def leer_ivm_media(ruta: str, anio: int) -> pd.DataFrame:
    """IVM de enseñanza media por establecimiento (hoja "Media" o "Ed. Media")."""
    hoja, t = _hoja_con_rbd(ruta, preferir=("media",))
    if "media" not in _norm(hoja):
        raise ValueError(f"{ruta}: no se encontró la hoja de enseñanza media.")
    c_rbd = next(c for c in t.columns if _re.fullmatch(r"(id_)?rbd", _norm(c).strip()))
    c_ivm = next(c for c in t.columns if _norm(c).strip() in ("ivm establecimiento", "ivm"))
    c_n = next((c for c in t.columns if "evaluado" in _norm(c)), None)
    out = pd.DataFrame({"rbd": pd.to_numeric(t[c_rbd], errors="coerce"),
                        "ivm_media": pd.to_numeric(t[c_ivm], errors="coerce")})
    if c_n:
        out["ivm_n"] = pd.to_numeric(t[c_n], errors="coerce")
    out = out.dropna(subset=["rbd", "ivm_media"]).assign(anio=anio)
    out["rbd"] = out["rbd"].astype(int)
    log.info("IVM media %d: hoja '%s', %d establecimientos.", anio, hoja, len(out))
    return out.drop_duplicates("rbd")


def panel_junaeb(carpeta, *, anios=(2015, 2016, 2017, 2018, 2022, 2023, 2024, 2025)) -> pd.DataFrame:
    """Lee todos los IVE e IVM de la carpeta para los años del panel.

    Ignora los años fuera del panel (2019-2021 y 2026), que no se usan.
    """
    from pathlib import Path
    carpeta = Path(carpeta)
    ive, ivm = [], []
    for f in sorted(carpeta.glob("*.xls*")):
        m = _re.search(r"(20\d{2})", f.name)
        if not m or int(m.group(1)) not in anios:
            continue
        anio = int(m.group(1)); nombre = _norm(f.name)
        try:
            if nombre.startswith("ive"):
                ive.append(leer_ive(str(f), anio))
            elif "ivm" in nombre and "establec" in nombre:
                ivm.append(leer_ivm_media(str(f), anio))
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo leer %s: %s", f.name, exc)
    p = pd.concat(ive) if ive else pd.DataFrame(columns=["rbd", "anio"])
    if ivm:
        p = p.merge(pd.concat(ivm), on=["rbd", "anio"], how="outer")
    return p.sort_values(["rbd", "anio"]).reset_index(drop=True)


def resumir_junaeb(panel: pd.DataFrame) -> pd.DataFrame:
    """Promedios por establecimiento en PRE (2015-2018) y POST (2022-2025) y su cambio.

    La covariable para los modelos es el promedio PRE; el cambio PRE→POST sirve
    para el contraste composicional (si la vulnerabilidad del establecimiento
    cambió, su caída podría reflejar un cambio de población).
    """
    panel = panel.copy()
    # Rango percentil dentro de cada año: el nivel medio del IVE sube entre
    # 2018 y 2022 para todos los establecimientos (de ~0,73 a ~0,81), sea por un
    # cambio de medición o de contexto. El cambio de rango aísla cuánto se movió
    # cada establecimiento respecto de los demás, sin ese desplazamiento común.
    if "ive" in panel.columns:
        panel["ive_rango"] = panel.groupby("anio")["ive"].rank(pct=True)
    cols = [c for c in ("ive", "ive_rango", "ivm_media", "matricula_total") if c in panel.columns]
    pre = panel[panel.anio.between(2015, 2018)].groupby("rbd")[cols].mean().add_suffix("_pre")
    post = panel[panel.anio.between(2022, 2025)].groupby("rbd")[cols].mean().add_suffix("_post")
    out = pre.join(post, how="outer")
    for c in cols:
        out[f"{c}_cambio"] = out[f"{c}_post"] - out[f"{c}_pre"]
    return out.reset_index()


def validar_junaeb(panel: pd.DataFrame, gse: pd.DataFrame) -> dict:
    """Controles de coherencia de los índices de JUNAEB.

    - Correlación con el GSE ordinal: debe ser negativa y fuerte (a mayor
      vulnerabilidad, menor GSE). Débil o positiva delata una columna mal leída.
    - Estabilidad entre años consecutivos: correlaciones altas indican medición
      consistente; una caída brusca entre dos años sugiere un cambio
      metodológico que obliga a no promediarlos.
    - Media por año: un salto abrupto del nivel también delata un cambio de
      metodología.
    """
    orden = {g: i for i, g in enumerate(("Bajo", "Medio bajo", "Medio", "Medio alto", "Alto"))}
    g = gse.assign(gse_ord=gse["cod_grupo_cat"].map(orden))[["rbd", "gse_ord"]]
    out = {}
    for c in [c for c in ("ive", "ivm_media") if c in panel.columns]:
        p = panel.dropna(subset=[c])
        w = p.pivot_table(index="rbd", columns="anio", values=c)
        anios = sorted(w.columns)
        out[c] = {
            "anios": anios,
            "media_por_anio": {int(a): round(float(w[a].mean()), 3) for a in anios},
            "r_entre_anios_consecutivos": {f"{a}-{b}": round(float(w[a].corr(w[b])), 3)
                                           for a, b in zip(anios, anios[1:])},
            "r_con_gse": round(float(p.groupby("rbd")[c].mean().rename("v").reset_index()
                                     .merge(g, on="rbd")[["v", "gse_ord"]].corr().iloc[0, 1]), 3),
        }
    return out


def ive_para_modelo(datos: pd.DataFrame, resumen: pd.DataFrame, *,
                    col_depe: str = "cod_depe1_cat") -> pd.DataFrame:
    """Incorpora el IVE previo al dataset, tratando la ausencia estructural.

    Los particulares pagados no participan de los programas de JUNAEB y no
    tienen IVE: no es un dato faltante al azar. Se les asigna 0 y un indicador
    ``ive_no_aplica``; como la dependencia ya está en los modelos, el
    coeficiente del IVE se estima dentro de los establecimientos subvencionados.
    Los subvencionados sin dato (cerrados o nuevos en el período) quedan como
    faltantes.
    """
    cols = [c for c in resumen.columns if c != "rbd"]
    out = datos.drop(columns=[c for c in cols if c in datos.columns]).merge(resumen, on="rbd", how="left")
    pagado = out[col_depe].eq("Particular pagado")
    out["ive_no_aplica"] = pagado.astype(int)
    if "ive_pre" in out.columns:
        out.loc[pagado, "ive_pre"] = 0.0
        sin = int((~pagado & out["ive_pre"].isna()).sum())
        log.info("IVE previo: %d pagados con 0 e indicador; %d subvencionados sin dato.",
                 int(pagado.sum()), sin)
    return out


# --------------------------------------------------------------------------- #
# Resumen de matrícula por establecimiento (datos abiertos MINEDUC)
# --------------------------------------------------------------------------- #
#
# El resumen NO desagrega por grado, solo por tipo de enseñanza agrupado.
# Codificación verificada con el archivo 2025 (el Liceo Politécnico de Arica
# suma MAT_ENS_7 = 862 y MAT_ENS_8 = 224, igual a su MAT_TOTAL de 1.086):
#   MAT_ENS_5: media humanístico-científica, jóvenes
#   MAT_ENS_7: media técnico-profesional y artística, jóvenes
# La matrícula de II medio se APROXIMA como un cuarto de la media de jóvenes,
# porque la media de jóvenes abarca cuatro grados (1.º a 4.º medio).

def procesar_resumen_matricula(ruta: str, anio: int) -> pd.DataFrame:
    """Matrícula de media de jóvenes y aproximación de la de II medio.

    Detecta el separador (el archivo 2025 usa coma; otros años pueden usar
    punto y coma) y las columnas por nombre.
    """
    with open(ruta, encoding="latin-1") as f:
        cab = f.readline()
    sep = ";" if cab.count(";") > cab.count(",") else ","
    t = pd.read_csv(ruta, sep=sep, encoding="latin-1", low_memory=False,
                    usecols=lambda c: _norm(c).strip('"') in ("rbd", "mat_ens_5", "mat_ens_7", "mat_total"))
    t.columns = [_norm(c).strip('"') for c in t.columns]
    for c in t.columns:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    out = pd.DataFrame({"rbd": t["rbd"].astype("Int64"), "anio": anio,
                        "matricula_media_jovenes": t["mat_ens_5"].fillna(0) + t["mat_ens_7"].fillna(0),
                        "matricula_total_mineduc": t["mat_total"]})
    out["matricula_2m_aprox"] = out["matricula_media_jovenes"] / 4.0
    out = out[out["matricula_media_jovenes"] > 0].dropna(subset=["rbd"])
    log.info("Resumen de matrícula %d: %d establecimientos con media de jóvenes (separador '%s').",
             anio, len(out), sep)
    return out.drop_duplicates("rbd")


def participacion_aproximada(ancho: pd.DataFrame, matricula: pd.DataFrame, *,
                             materia: str = "mate") -> pd.DataFrame:
    """Evaluados de II medio sobre la matrícula de II medio aproximada.

    Como la matrícula de II medio se aproxima con un cuarto de la media de
    jóvenes, el NIVEL de la tasa tiene un sesgo propio de cada establecimiento
    (cursos de distinto tamaño, media incompleta). Ese sesgo es estable en el
    tiempo y se cancela en el CAMBIO de la tasa entre períodos, que es lo que
    usa el contraste de la alternativa (ii).
    """
    m = matricula.rename(columns={"matricula_2m_aprox": "matricula_2m"})[["rbd", "anio", "matricula_2m"]]
    return tasa_participacion(ancho, m, materia=materia)


# --------------------------------------------------------------------------- #
# Pobreza comunal por áreas pequeñas (Ministerio de Desarrollo Social y Familia)
# --------------------------------------------------------------------------- #

def leer_pobreza_sae(ruta: str) -> pd.DataFrame:
    """Tasa de pobreza por ingresos y población proyectada, por comuna (CUT).

    Fuente: base de datos de pobreza comunal (estimación para áreas pequeñas,
    SAE), https://bidat.gob.cl. Además de la tasa trae la población proyectada
    de cada comuna, que se usa como covariable de tamaño comunal.

    Se conserva el tipo de estimación: las comunas con estimación solo
    "Sintética" no tienen muestra CASEN propia y su tasa es menos precisa, lo
    que permite un análisis de sensibilidad que las excluya.
    """
    _, t = _hoja_con_clave(ruta, "codigo")
    c_cod = detectar_columna(t.columns, "codigo")
    c_tasa = detectar_columna(t.columns, "porcentaje", "pobreza")
    c_pob = detectar_columna(t.columns, "proyecc", obligatoria=False)
    c_npob = detectar_columna(t.columns, "numero", "pobreza", obligatoria=False)
    c_inf = detectar_columna(t.columns, "inferior", obligatoria=False)
    c_sup = detectar_columna(t.columns, "superior", obligatoria=False)
    c_tipo = (detectar_columna(t.columns, "tipo", "estimacion", obligatoria=False)
              or detectar_columna(t.columns, "metodologia", obligatoria=False))
    out = pd.DataFrame({"cut": pd.to_numeric(t[c_cod], errors="coerce"),
                        "pobreza_comunal": pd.to_numeric(t[c_tasa], errors="coerce")})
    if c_pob:
        out["poblacion_comunal"] = pd.to_numeric(t[c_pob], errors="coerce")
    elif c_npob:
        # Los archivos anteriores a 2020 no traen la población proyectada, pero
        # sí el número de personas en pobreza y su porcentaje: la población es
        # exactamente el cociente entre ambos.
        tasa = out["pobreza_comunal"] / (100.0 if out["pobreza_comunal"].max() > 1.5 else 1.0)
        out["poblacion_comunal"] = pd.to_numeric(t[c_npob], errors="coerce") / tasa
        log.info("Población comunal derivada del número de pobres y la tasa.")
    if c_inf and c_sup:
        out["pobreza_ic_ancho"] = (pd.to_numeric(t[c_sup], errors="coerce")
                                   - pd.to_numeric(t[c_inf], errors="coerce"))
    if c_tipo:
        tipo = t[c_tipo].astype(str).map(_norm)
        out["pobreza_solo_sintetica"] = (tipo.str.contains("sintetic") & ~tipo.str.contains("direct")).astype(int)
    out = out.dropna(subset=["cut", "pobreza_comunal"])
    if out["pobreza_comunal"].max() > 1.5:
        out["pobreza_comunal"] /= 100.0
    out["cut"] = out["cut"].astype(int)
    log.info("Pobreza comunal: %d comunas; tasa mediana %.3f.", len(out),
             float(out["pobreza_comunal"].median()))
    return out


def _hoja_con_clave(ruta: str, clave: str):
    """Primera hoja con una fila de encabezado que contenga ``clave``."""
    x = pd.ExcelFile(ruta)
    for h in x.sheet_names:
        crudo = pd.read_excel(x, sheet_name=h, header=None, nrows=40)
        for i in range(len(crudo)):
            if any(_norm(v).strip() == clave for v in crudo.iloc[i]):
                t = pd.read_excel(x, sheet_name=h, header=i)
                t.columns = [str(c).strip() for c in t.columns]
                return h, t
    raise ValueError(f"{ruta}: no se encontró el encabezado '{clave}'.")


def validar_pobreza(datos: pd.DataFrame, pobreza: pd.DataFrame, *,
                    col_comuna: str = "comuna_dir", col_gse: str = "cod_grupo_cat") -> dict:
    """Cobertura de los códigos comunales y coherencia con el GSE.

    La proporción de establecimientos de GSE Bajo o Medio bajo en una comuna
    debe correlacionar positivamente con su tasa de pobreza.
    """
    d = datos.assign(cut=pd.to_numeric(datos[col_comuna], errors="coerce"))
    cobertura = float(d["cut"].isin(pobreza["cut"]).mean())
    sin = sorted(d.loc[~d["cut"].isin(pobreza["cut"]), "cut"].dropna().astype(int).unique().tolist())
    vul = (d.assign(v=d[col_gse].isin(["Bajo", "Medio bajo"]).astype(float))
            .groupby("cut")["v"].mean().rename("prop_gse_vulnerable"))
    j = pobreza.set_index("cut").join(vul, how="inner")
    return {"cobertura_establecimientos": round(cobertura, 3),
            "comunas_sin_codigo": sin[:20],
            "r_pobreza_vs_gse_vulnerable": round(float(j["pobreza_comunal"].corr(j["prop_gse_vulnerable"])), 3),
            "comunas_comparadas": len(j)}


# --------------------------------------------------------------------------- #
# Pobreza comunal en varios años y superficie comunal
# --------------------------------------------------------------------------- #

#: Variantes ortográficas de nombres comunales entre fuentes oficiales. Se
#: aplican a ambos lados del cruce, por lo que no importa cuál use cada fuente.
_VARIANTES_COMUNA: dict[str, str] = {
    "aisen": "aysen", "coihaique": "coyhaique", "llayllay": "llaillay",
    "marchigue": "marchihue", "paihuano": "paiguano", "trehuaco": "treguaco",
    "cabodehornosexnavarino": "cabodehornos",
}


def _clave_nombre(s) -> str:
    """Nombre de comuna normalizado para cruzar fuentes.

    Quita tildes, puntuación y espacios; elimina un sufijo entre paréntesis (la
    BCN escribe "Arica (Región de Arica y Parinacota)"), y unifica variantes
    ortográficas conocidas.
    """
    base = _re.sub(r"\(.*?\)", "", str(s))
    clave = _re.sub(r"[^a-z]", "", _norm(base))
    return _VARIANTES_COMUNA.get(clave, clave)


def panel_pobreza(carpeta) -> pd.DataFrame:
    """Lee todos los ``sae_ingresos_<año>.xlsx`` y armoniza los códigos comunales.

    Las estimaciones anteriores a 2018 usan los códigos de la antigua provincia
    de Ñuble (84xx), que pasaron a la Región de Ñuble (16xxx). La equivalencia
    se deduce cruzando por nombre de comuna contra el archivo más reciente, en
    lugar de escribirla a mano.

    Nota metodológica: la estimación 2024 usa la nueva medición de pobreza, de
    modo que su nivel no es comparable con los años anteriores. Por eso la
    covariable se construye con los años previos y no se usa el cambio.
    """
    from pathlib import Path
    archivos = sorted(Path(carpeta).glob("sae_ingresos_*.xls*"))
    tablas = {}
    for f in archivos:
        anio = int(_re.search(r"(20\d{2})", f.name).group(1))
        try:
            _, t = _hoja_con_clave(str(f), "codigo")
            c_nom = detectar_columna(t.columns, "comuna", obligatoria=False) or \
                detectar_columna(t.columns, "nombre", obligatoria=False)
            base = leer_pobreza_sae(str(f))
            if c_nom:
                nombres = t[[detectar_columna(t.columns, "codigo"), c_nom]].copy()
                nombres.columns = ["cut", "nombre"]
                nombres["cut"] = pd.to_numeric(nombres["cut"], errors="coerce")
                base = base.merge(nombres.dropna(subset=["cut"]).astype({"cut": int}),
                                  on="cut", how="left")
            tablas[anio] = base.assign(anio=anio)
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo leer %s: %s", f.name, exc)
    if not tablas:
        return pd.DataFrame()
    ref = tablas[max(tablas)]
    por_nombre = dict(zip(ref["nombre"].map(_clave_nombre), ref["cut"])) if "nombre" in ref else {}
    salida = []
    for anio, t in tablas.items():
        if por_nombre and "nombre" in t:
            nuevo = t["nombre"].map(_clave_nombre).map(por_nombre)
            cambiados = int((nuevo.notna() & (nuevo != t["cut"])).sum())
            if cambiados:
                log.info("Pobreza %d: %d códigos comunales armonizados por nombre.", anio, cambiados)
            t = t.assign(cut=nuevo.fillna(t["cut"]).astype(int))
        salida.append(t)
    return pd.concat(salida, ignore_index=True)


def _a_numero_regional(serie: pd.Series) -> pd.Series:
    """Convierte números con coma o punto decimal sin multiplicarlos por error.

    Si algún valor trae coma, la coma es el decimal y el punto, separador de
    miles ("1.234,5"). Si no hay comas, el punto es el decimal ("22.4"). Borrar
    el punto en ese segundo caso multiplicaría el valor por diez o más.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    txt = serie.astype(str).str.strip()
    if txt.str.contains(",", regex=False).any():
        txt = txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def leer_superficie(ruta: str, referencia: pd.DataFrame) -> pd.DataFrame:
    """Superficie comunal (km²), con el código comunal asignado por nombre.

    Tolera los formatos de exportación más comunes: largo (comuna, año, valor)
    o ancho (comuna y una columna por año). Si hay varios años se usa el más
    reciente. ``referencia`` debe traer ``cut`` y ``nombre`` (por ejemplo, la
    pobreza más reciente), para asignar el código a cada nombre.
    """
    with open(ruta, encoding="latin-1") as f:
        cab = f.readline()
    sep = ";" if cab.count(";") >= cab.count(",") else ","
    t = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            t = pd.read_csv(ruta, sep=sep, encoding=enc); break
        except UnicodeDecodeError:
            continue
    t.columns = [str(c).strip() for c in t.columns]
    c_nom = next((c for c in t.columns if any(p in _norm(c) for p in ("comuna", "unidad", "territor", "nombre"))), t.columns[0])
    anios = [c for c in t.columns if _re.fullmatch(r"(19|20)\d{2}", str(c).strip())]
    if anios:                                             # formato ancho
        valor = t[sorted(anios)[-1]]
    else:                                                 # formato largo
        c_val = next((c for c in t.columns if any(p in _norm(c) for p in ("superficie", "valor", "km"))), None)
        c_anio = next((c for c in t.columns if "ano" in _norm(c) or "periodo" in _norm(c)), None)
        if c_anio is not None:
            t = t.sort_values(c_anio).groupby(c_nom, as_index=False).last()
        valor = t[c_val]
    # La BCN agrega la región entre paréntesis ("Arica (Región de Arica y
    # Parinacota)"); se elimina antes de cruzar por nombre.
    sup = pd.DataFrame({"nombre": t[c_nom].astype(str).str.replace(r"\s*\(.*?\)\s*", "", regex=True),
                        "superficie_km2": _a_numero_regional(valor)})
    mapa = dict(zip(referencia["nombre"].map(_clave_nombre), referencia["cut"]))
    sup["cut"] = sup["nombre"].map(_clave_nombre).map(mapa)
    sin = sup[sup["cut"].isna() & sup["superficie_km2"].notna()]["nombre"].tolist()
    if sin:
        log.warning("Superficie: %d nombres sin código: %s", len(sin), sin)
    sup = sup.dropna(subset=["cut", "superficie_km2"]).astype({"cut": int})
    log.info("Superficie: %d comunas con código.", len(sup))
    return sup[["cut", "superficie_km2"]].drop_duplicates("cut")


def validar_superficie(sup: pd.DataFrame) -> dict:
    """Controles con valores conocidos: la comuna de Santiago mide unos 22 km²,
    y ninguna comuna continental supera unos 50.000 km² (las mayores están en
    Magallanes y Aysén). Valores fuera de rango delatan un problema de
    separador decimal o de unidad."""
    santiago = sup.loc[sup["cut"] == 13101, "superficie_km2"]
    return {"comunas": len(sup),
            "santiago_km2": round(float(santiago.iloc[0]), 1) if len(santiago) else None,
            "santiago_plausible": bool(len(santiago) and 15 <= santiago.iloc[0] <= 30),
            "max_km2": round(float(sup["superficie_km2"].max()), 1),
            "min_km2": round(float(sup["superficie_km2"].min()), 2)}


def contraste_participacion(participacion: pd.DataFrame, gse: pd.DataFrame, *,
                            anio_base: int = 2018, anio_post: int = 2022) -> dict:
    """Contraste de la alternativa (ii): ¿cayó más la participación en los GSE bajos?

    Usa el cambio de la tasa dentro de cada establecimiento, que cancela el
    sesgo propio de la aproximación de la matrícula. Devuelve el cambio por
    GSE y la diferencia de cada grupo respecto del Bajo, con errores robustos.
    """
    import statsmodels.formula.api as smf
    w = participacion.pivot_table(index="rbd", columns="anio", values="tasa_participacion")
    if anio_base not in w or anio_post not in w:
        raise ValueError("Faltan los años necesarios para el contraste.")
    d = (w[anio_post] - w[anio_base]).rename("cambio").reset_index().merge(gse, on="rbd").dropna()
    d = d.rename(columns={"cod_grupo_cat": "gse"})
    orden = ["Bajo", "Medio bajo", "Medio", "Medio alto", "Alto"]
    por_grupo = d.groupby("gse")["cambio"].agg(["count", "median", "mean"]).reindex(orden).round(3)
    m = smf.ols('cambio ~ C(gse, Treatment("Bajo"))', data=d).fit(cov_type="HC3")
    dif = {}
    for g in orden[1:]:
        k = f'C(gse, Treatment("Bajo"))[T.{g}]'
        if k in m.params:
            dif[g] = [round(float(m.params[k]), 3)] + [round(float(v), 3) for v in m.conf_int().loc[k]]
    return {"por_gse": por_grupo, "diferencia_vs_bajo_[est,ic_inf,ic_sup]": dif}
