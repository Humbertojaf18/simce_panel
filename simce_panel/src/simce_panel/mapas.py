"""
mapas.py
========

Representación cartográfica de los resultados sobre un mapa base real.

Dos salidas complementarias
---------------------------
- :func:`mapa_interactivo` produce un mapa HTML (folium/Leaflet) con mosaicos
  de OpenStreetMap o CartoDB. Los mosaicos los descarga **el navegador** al
  abrir el archivo, de modo que funciona en cualquier equipo con conexión y
  sirve para explorar y para compartir.
- :func:`mapa_estatico` produce una figura PNG con mapa base, apta para
  insertar en el documento de la tesis. Aquí los mosaicos los descarga
  **Python** al generar la figura, por lo que requiere conexión en el momento
  de ejecutarla.

Resguardo de identificabilidad
------------------------------
Ninguna función de este módulo expone el RBD ni el nombre del establecimiento
en etiquetas, leyendas o ventanas emergentes. El compromiso metodológico de la
tesis es mostrar el patrón espacial sin permitir construir un listado de
escuelas por desempeño, y un mapa con buscador por nombre lo haría trivial.
Cuando se necesita contexto se muestra la comuna, que agrupa a varios
establecimientos. Adicionalmente, :func:`desplazar_coordenadas` permite añadir
un desplazamiento aleatorio controlado para las figuras publicables.

Funciones principales
---------------------
- :func:`desplazar_coordenadas` : ruido espacial controlado para publicación.
- :func:`mapa_interactivo`      : mapa HTML con mosaicos reales.
- :func:`mapa_conglomerados`    : mapa HTML de categorías LISA.
- :func:`mapa_estatico`         : figura PNG con mapa base.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Centro aproximado de Chile continental, para encuadrar el mapa.
CENTRO_CHILE: tuple[float, float] = (-35.5, -71.3)

#: Mosaicos cartográficos disponibles, todos **gratuitos y sin API key**.
#:
#: Nota sobre CartoDB: sus mosaicos (Positron, Dark Matter) pasaron a requerir
#: una clave de API, por lo que sin ella el mapa se rellena con la marca de
#: agua "API KEY REQUIRED". Por eso no se usan por defecto. Si se dispone de
#: una clave gratuita de Carto, puede pasarse en ``api_key_carto``.
#:
#: Para superponer datos conviene un fondo de baja saturación, que no compita
#: con los colores del dato: ``gris`` (Esri Light Gray) cumple ese papel y es
#: el equivalente más cercano a Positron.
MOSAICOS: dict[str, dict[str, str]] = {
    "gris": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"),
        "attr": "Tiles &copy; Esri",
        "nombre": "Gris claro (Esri)",
    },
    "calles": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "&copy; OpenStreetMap contributors",
        "nombre": "Calles (OpenStreetMap)",
    },
    "satelital": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "attr": "Tiles &copy; Esri",
        "nombre": "Satelital (Esri)",
    },
    "relieve": {
        "url": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "&copy; OpenTopoMap (CC-BY-SA)",
        "nombre": "Relieve (OpenTopoMap)",
    },
}

#: Mosaico por defecto: fondo claro, legible y sin clave.
MOSAICO_DEFECTO: str = "gris"


def _crear_mapa(mosaico: str, api_key_carto: str | None = None):
    """Crea el mapa base de folium con el mosaico indicado.

    Si se entrega ``api_key_carto`` se usan los mosaicos Positron de CartoDB,
    que requieren clave; en caso contrario se emplea uno de los proveedores
    libres de :data:`MOSAICOS`.
    """
    import folium

    if api_key_carto:
        url = ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
               f"?api_key={api_key_carto}")
        return folium.Map(location=CENTRO_CHILE, zoom_start=4, tiles=url,
                          attr="&copy; OpenStreetMap contributors &copy; CARTO",
                          control_scale=True)

    cfg = MOSAICOS.get(mosaico, MOSAICOS[MOSAICO_DEFECTO])
    m = folium.Map(location=CENTRO_CHILE, zoom_start=4, tiles=None,
                   control_scale=True)
    folium.TileLayer(tiles=cfg["url"], attr=cfg["attr"], name=cfg["nombre"],
                     max_zoom=18).add_to(m)
    # Se ofrecen los demás fondos como capas alternativas, seleccionables
    # desde el control de capas del propio mapa.
    for clave, otro in MOSAICOS.items():
        if clave == mosaico:
            continue
        folium.TileLayer(tiles=otro["url"], attr=otro["attr"],
                         name=otro["nombre"], max_zoom=18, show=False).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m

#: Colores de los conglomerados LISA.
COLORES_LISA: dict[str, str] = {
    "Alto-Alto": "#d7191c",
    "Bajo-Bajo": "#2c7bb6",
    "Alto-Bajo": "#fdae61",
    "Bajo-Alto": "#abd9e9",
    "no significativo": "#cccccc",
}


def desplazar_coordenadas(
    df: pd.DataFrame,
    *,
    radio_m: float = 300.0,
    col_lat: str = "latitud",
    col_lon: str = "longitud",
    semilla: int = 42,
) -> pd.DataFrame:
    """Añade un desplazamiento aleatorio a las coordenadas.

    Técnica habitual en cartografía con datos sensibles: conserva el patrón
    espacial a la escala de análisis (varios kilómetros) pero impide
    identificar el establecimiento exacto a partir del punto. Con un radio de
    300 metros, en zonas urbanas densas el punto puede caer sobre manzanas
    vecinas, mientras que a escala regional el mapa es indistinguible.

    Parameters
    ----------
    df : pandas.DataFrame
    radio_m : float, optional
        Radio máximo del desplazamiento, en metros.
    col_lat, col_lon : str, optional
    semilla : int, optional
        Fija el generador para que el mapa sea reproducible.

    Returns
    -------
    pandas.DataFrame
        Copia con las coordenadas desplazadas.
    """
    rng = np.random.default_rng(semilla)
    out = df.copy()
    n = len(out)
    # Desplazamiento uniforme dentro de un disco.
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radio_m * np.sqrt(rng.uniform(0, 1, n))
    grados_lat = rad / 111_320.0
    lat = out[col_lat].to_numpy(dtype="float64")
    grados_lon = rad / (111_320.0 * np.cos(np.radians(lat)))
    out[col_lat] = lat + grados_lat * np.sin(ang)
    out[col_lon] = out[col_lon].to_numpy(dtype="float64") + grados_lon * np.cos(ang)
    log.info("Coordenadas desplazadas hasta %.0f m (semilla %d).", radio_m, semilla)
    return out


def _escala_divergente(valores: pd.Series, limite: float | None = None):
    """Devuelve una función valor -> color hexadecimal, centrada en cero."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    v = pd.to_numeric(valores, errors="coerce")
    if limite is None:
        limite = float(np.nanpercentile(np.abs(v.dropna()), 95)) or 1.0
    norma = mcolors.TwoSlopeNorm(vmin=-limite, vcenter=0.0, vmax=limite)
    mapa = cm.ScalarMappable(norm=norma, cmap="RdYlBu")

    def color(x):
        if pd.isna(x):
            return "#999999"
        return mcolors.to_hex(mapa.to_rgba(float(x)))

    return color, limite


def mapa_interactivo(
    df: pd.DataFrame,
    columna: str,
    *,
    col_lat: str = "latitud",
    col_lon: str = "longitud",
    col_contexto: str | None = "comuna_dir",
    mosaico: str = MOSAICO_DEFECTO,
    api_key_carto: str | None = None,
    radio_punto: int = 3,
    limite_color: float | None = None,
    titulo: str = "",
    desplazar: bool = False,
    radio_desplazamiento_m: float = 300.0,
    ruta_html: str | None = None,
):
    """Mapa interactivo de una variable continua sobre mosaicos reales.

    Parameters
    ----------
    df : pandas.DataFrame
        Debe traer coordenadas y la columna a representar.
    columna : str
        Variable continua; se colorea con una escala divergente centrada en 0.
    col_lat, col_lon : str, optional
    col_contexto : str | None, optional
        Columna mostrada al pasar el cursor, como referencia territorial. Debe
        ser una unidad agregada (comuna, región): nunca el RBD ni el nombre.
    mosaico : str, optional
        Clave de :data:`MOSAICOS`: ``gris``, ``calles``, ``satelital`` o
        ``relieve``. Todos son gratuitos y no requieren clave.
    api_key_carto : str | None, optional
        Clave gratuita de CartoDB, si se prefiere el fondo Positron.
    radio_punto : int, optional
    limite_color : float | None, optional
        Valor absoluto que satura la escala. Si es ``None`` se usa el
        percentil 95 del valor absoluto, para que unos pocos casos extremos no
        aplasten el resto de la paleta.
    titulo : str, optional
    desplazar : bool, optional
        Aplica :func:`desplazar_coordenadas` antes de dibujar.
    radio_desplazamiento_m : float, optional
    ruta_html : str | None, optional
        Si se indica, guarda el mapa en esa ruta.

    Returns
    -------
    folium.Map
        En Jupyter se muestra directamente al devolverlo.
    """
    import folium

    d = df.dropna(subset=[col_lat, col_lon, columna]).copy()
    if desplazar:
        d = desplazar_coordenadas(d, radio_m=radio_desplazamiento_m,
                                  col_lat=col_lat, col_lon=col_lon)

    color_de, limite = _escala_divergente(d[columna], limite_color)

    m = _crear_mapa(mosaico, api_key_carto)

    for _, fila in d.iterrows():
        valor = fila[columna]
        contexto = ""
        if col_contexto and col_contexto in d.columns and pd.notna(fila[col_contexto]):
            contexto = f"<br>{col_contexto}: {fila[col_contexto]}"
        folium.CircleMarker(
            location=(float(fila[col_lat]), float(fila[col_lon])),
            radius=radio_punto,
            color=None,
            fill=True,
            fill_color=color_de(valor),
            fill_opacity=0.75,
            weight=0,
            # Sin RBD ni nombre: solo el valor y la unidad territorial.
            tooltip=f"{columna}: {valor:.1f}{contexto}",
        ).add_to(m)

    _agregar_leyenda_continua(m, columna, limite, titulo)

    if ruta_html:
        m.save(ruta_html)
        log.info("Mapa guardado en %s", ruta_html)
    return m


def mapa_conglomerados(
    df: pd.DataFrame,
    columna_cluster: str,
    *,
    col_lat: str = "latitud",
    col_lon: str = "longitud",
    mosaico: str = MOSAICO_DEFECTO,
    api_key_carto: str | None = None,
    solo_significativos: bool = False,
    titulo: str = "",
    ruta_html: str | None = None,
):
    """Mapa interactivo de las categorías de conglomerado LISA.

    Parameters
    ----------
    df : pandas.DataFrame
    columna_cluster : str
        Columna con las etiquetas de conglomerado.
    solo_significativos : bool, optional
        Si ``True`` omite los puntos no significativos, útil cuando estos
        dominan visualmente el mapa.
    ruta_html : str | None, optional

    Returns
    -------
    folium.Map
    """
    import folium

    d = df.dropna(subset=[col_lat, col_lon, columna_cluster]).copy()
    if solo_significativos:
        d = d[d[columna_cluster] != "no significativo"]
        if d.empty:
            log.warning("No hay conglomerados significativos que dibujar.")

    m = _crear_mapa(mosaico, api_key_carto)

    for _, fila in d.iterrows():
        etiqueta = str(fila[columna_cluster])
        folium.CircleMarker(
            location=(float(fila[col_lat]), float(fila[col_lon])),
            radius=3 if etiqueta == "no significativo" else 5,
            fill=True,
            fill_color=COLORES_LISA.get(etiqueta, "#999999"),
            fill_opacity=0.35 if etiqueta == "no significativo" else 0.9,
            weight=0,
            tooltip=etiqueta,
        ).add_to(m)

    presentes = [e for e in COLORES_LISA if e in set(d[columna_cluster])]
    _agregar_leyenda_categorica(m, presentes, titulo)

    if ruta_html:
        m.save(ruta_html)
        log.info("Mapa guardado en %s", ruta_html)
    return m


def _agregar_leyenda_continua(m, columna: str, limite: float, titulo: str) -> None:
    """Inserta una leyenda de escala divergente como HTML sobre el mapa."""
    html = f"""
    <div style="position: fixed; bottom: 30px; left: 20px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #999;
                border-radius: 4px; font-family: sans-serif; font-size: 12px;">
      <b>{titulo or columna}</b><br>
      <div style="margin-top:6px; width: 180px; height: 12px;
                  background: linear-gradient(to right, #2c7bb6, #ffffbf, #d7191c);
                  border: 1px solid #bbb;"></div>
      <div style="display:flex; justify-content:space-between; width:180px;">
        <span>-{limite:.0f}</span><span>0</span><span>+{limite:.0f}</span>
      </div>
    </div>"""
    m.get_root().html.add_child(__import__("folium").Element(html))


def _agregar_leyenda_categorica(m, etiquetas: list[str], titulo: str) -> None:
    """Inserta una leyenda de categorías LISA sobre el mapa."""
    filas = "".join(
        f'<div><span style="display:inline-block;width:12px;height:12px;'
        f'background:{COLORES_LISA[e]};margin-right:6px;'
        f'border:1px solid #999;"></span>{e}</div>'
        for e in etiquetas)
    html = f"""
    <div style="position: fixed; bottom: 30px; left: 20px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #999;
                border-radius: 4px; font-family: sans-serif; font-size: 12px;">
      <b>{titulo or 'Conglomerados'}</b><br>{filas}
    </div>"""
    m.get_root().html.add_child(__import__("folium").Element(html))


# --------------------------------------------------------------------------- #
# Mapa base estático: descarga propia de mosaicos (sin contextily)
# --------------------------------------------------------------------------- #
#
# Por qué no se usa contextily
# ----------------------------
# contextily depende de rasterio para reproyectar los mosaicos, y esa cadena
# de dependencias es frágil en Windows: puede fallar aunque la red funcione
# (se verificó que el navegador del mismo equipo descarga los mosaicos sin
# problemas). Aquí los mosaicos se descargan directamente, igual que lo hace
# un navegador, y se dibujan en la proyección Web Mercator en la que vienen,
# sin reproyectar. Las marcas de los ejes se rotulan en grados.

_RADIO_MERCATOR = 6378137.0


def _a_mercator(lat, lon):
    """Convierte grados a coordenadas Web Mercator (metros)."""
    lat = np.clip(np.asarray(lat, dtype="float64"), -85.0511, 85.0511)
    x = _RADIO_MERCATOR * np.radians(np.asarray(lon, dtype="float64"))
    y = _RADIO_MERCATOR * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def _desde_mercator_lat(y):
    """Latitud en grados a partir de la coordenada Y de Web Mercator."""
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y) / _RADIO_MERCATOR)) - np.pi / 2)


def _tile_xy(lat, lon, z):
    """Índices (x, y) del mosaico XYZ que contiene el punto, a zoom ``z``."""
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = np.radians(lat)
    y = int((1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / np.pi) / 2 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _limites_tile_mercator(x, y, z):
    """Extensión (xmin, xmax, ymin, ymax) de un mosaico, en Web Mercator."""
    n = 2 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    lat_min = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
    x0, y0 = _a_mercator(lat_min, lon_min)
    x1, y1 = _a_mercator(lat_max, lon_max)
    return float(x0), float(x1), float(y0), float(y1)


def _descargar_tile(url: str, timeout: float = 15.0) -> bytes:
    """Descarga un mosaico declarando un User-Agent, como exige OSM."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def componer_fondo(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    *,
    zoom: int = 6,
    mosaico: str = MOSAICO_DEFECTO,
    descargar=None,
):
    """Descarga y ensambla los mosaicos que cubren un rectángulo geográfico.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : float
        Rectángulo a cubrir, en grados.
    zoom : int, optional
        Nivel de zoom. 6 da buena resolución para Chile completo con unos 50
        mosaicos; más alto multiplica las descargas por cuatro en cada nivel.
    mosaico : str, optional
        Clave de :data:`MOSAICOS`.
    descargar : callable, optional
        Función ``url -> bytes``. Por defecto descarga por HTTP; se puede
        sustituir para pruebas sin conexión.

    Returns
    -------
    tuple
        ``(imagen, extension)``: arreglo RGB y ``(xmin, xmax, ymin, ymax)``
        en Web Mercator, listo para ``imshow``.
    """
    import io
    from PIL import Image

    descargar = descargar or _descargar_tile
    plantilla = MOSAICOS.get(mosaico, MOSAICOS[MOSAICO_DEFECTO])["url"]

    x0, y0 = _tile_xy(lat_max, lon_min, zoom)   # esquina superior izquierda
    x1, y1 = _tile_xy(lat_min, lon_max, zoom)   # esquina inferior derecha
    ancho, alto = (x1 - x0 + 1), (y1 - y0 + 1)
    log.info("Descargando %d mosaicos (zoom %d, proveedor '%s').",
             ancho * alto, zoom, mosaico)

    lienzo = Image.new("RGB", (ancho * 256, alto * 256), (230, 230, 230))
    fallidos = 0
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            url = plantilla.format(z=zoom, x=tx, y=ty)
            try:
                tile = Image.open(io.BytesIO(descargar(url))).convert("RGB")
                if tile.size != (256, 256):
                    tile = tile.resize((256, 256))
                lienzo.paste(tile, ((tx - x0) * 256, (ty - y0) * 256))
            except Exception as exc:  # noqa: BLE001
                fallidos += 1
                if fallidos == 1:
                    log.warning("Falló la descarga de %s: %s: %s", url,
                                type(exc).__name__, str(exc)[:80])
    if fallidos:
        log.warning("%d de %d mosaicos no se pudieron descargar.",
                    fallidos, ancho * alto)
        if fallidos == ancho * alto:
            raise ConnectionError("No se descargó ningún mosaico.")

    xmin, _, _, ymax = _limites_tile_mercator(x0, y0, zoom)
    _, xmax, ymin, _ = _limites_tile_mercator(x1, y1, zoom)
    return np.asarray(lienzo), (xmin, xmax, ymin, ymax)


def mapa_estatico(
    df: pd.DataFrame,
    columna: str,
    *,
    col_lat: str = "latitud",
    col_lon: str = "longitud",
    recorte: tuple[float, float, float, float] | None = None,
    limite_color: float | None = None,
    titulo: str = "",
    mosaico: str = MOSAICO_DEFECTO,
    zoom: int = 6,
    figsize: tuple[float, float] = (6, 11),
    ruta_png: str | None = None,
    dpi: int = 200,
    credito: bool = True,
    descargar=None,
):
    """Figura PNG con mapa base, para insertar en el documento de la tesis.

    Los mosaicos se descargan directamente (ver :func:`componer_fondo`) y se
    dibujan en Web Mercator; los ejes se rotulan en grados. Si la descarga
    falla por completo, la figura se genera igual sin fondo y se advierte la
    causa concreta.

    Parameters
    ----------
    df : pandas.DataFrame
    columna : str
    recorte : tuple | None, optional
        ``(lon_min, lon_max, lat_min, lat_max)``. Por defecto encuadra Chile
        continental, para que Rapa Nui no comprima la escala.
    limite_color : float | None, optional
    titulo : str, optional
    mosaico : str, optional
        Clave de :data:`MOSAICOS`. Para figuras impresas, ``gris`` es el más
        legible.
    zoom : int, optional
    figsize : tuple[float, float], optional
    ruta_png : str | None, optional
    dpi : int, optional
    credito : bool, optional
        Añade la atribución del mapa base en la esquina, obligatoria por las
        licencias de los proveedores.
    descargar : callable, optional
        Sustituye la descarga HTTP (útil para pruebas sin conexión).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter

    d = df.dropna(subset=[col_lat, col_lon, columna]).copy()
    if recorte is None:
        recorte = (-76.0, -66.0, -56.0, -17.0)   # Chile continental
    lon_min, lon_max, lat_min, lat_max = recorte
    d = d[(d[col_lon].between(lon_min, lon_max))
          & (d[col_lat].between(lat_min, lat_max))]

    _, limite = _escala_divergente(d[columna], limite_color)
    norma = mcolors.TwoSlopeNorm(vmin=-limite, vcenter=0.0, vmax=limite)

    fig, ax = plt.subplots(figsize=figsize)

    fondo_ok = False
    try:
        img, ext = componer_fondo(lat_min, lat_max, lon_min, lon_max,
                                  zoom=zoom, mosaico=mosaico,
                                  descargar=descargar)
        ax.imshow(img, extent=ext, origin="upper", zorder=0,
                  interpolation="bilinear")
        fondo_ok = True
    except Exception as exc:  # noqa: BLE001
        log.warning("Sin mapa base (%s: %s). Ejecuta "
                    "sp.mapas.diagnosticar_mosaicos() para ver la causa.",
                    type(exc).__name__, str(exc)[:100])

    px, py = _a_mercator(d[col_lat].to_numpy(), d[col_lon].to_numpy())
    puntos = ax.scatter(px, py, c=d[columna], cmap="RdYlBu", norm=norma,
                        s=7, alpha=0.9, linewidths=0.2, edgecolors="#555555",
                        zorder=3)

    bx0, by0 = _a_mercator(lat_min, lon_min)
    bx1, by1 = _a_mercator(lat_max, lon_max)
    ax.set_xlim(float(bx0), float(bx1))
    ax.set_ylim(float(by0), float(by1))
    ax.set_aspect("equal")

    # Ejes rotulados en grados aunque el dibujo esté en metros Mercator.
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{np.degrees(v / _RADIO_MERCATOR):.0f}°"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{float(_desde_mercator_lat(v)):.0f}°"))
    ax.tick_params(labelsize=7)

    ax.set_title(titulo or columna, fontsize=11)
    if credito and fondo_ok:
        attr = MOSAICOS.get(mosaico, MOSAICOS[MOSAICO_DEFECTO])["attr"]
        attr = attr.replace("&copy;", "©")
        ax.text(0.99, 0.005, f"Mapa base: {attr}", transform=ax.transAxes,
                fontsize=6, ha="right", va="bottom", color="#444444",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    fig.colorbar(puntos, ax=ax, shrink=0.55, label=columna)
    fig.tight_layout()

    if ruta_png:
        fig.savefig(ruta_png, dpi=dpi, bbox_inches="tight")
        log.info("Figura guardada en %s", ruta_png)
    return fig


# --------------------------------------------------------------------------- #
# Diagnóstico de conectividad con los servidores de mosaicos
# --------------------------------------------------------------------------- #

#: Mosaico de prueba sobre Santiago a zoom 4 (x=4, y=9 en esquema XYZ).
_MOSAICO_PRUEBA = {"z": 4, "x": 4, "y": 9}

#: Identificación del cliente. OpenStreetMap rechaza con error 403 las
#: descargas que no declaran un User-Agent propio, como exige su política de
#: uso de mosaicos.
_USER_AGENT = "simce_panel/1.0 (tesis MDS UDLA; uso academico)"


def diagnosticar_mosaicos(timeout: float = 10.0) -> pd.DataFrame:
    """Prueba la descarga de un mosaico desde cada proveedor.

    Distingue las dos causas posibles de un mapa sin fondo cartográfico:

    - **Bloqueo de red**: el equipo no alcanza el servidor (proxy, firewall,
      antivirus o red institucional). Afecta a todas las salidas.
    - **Rechazo del servidor**: el servidor responde con error, por ejemplo
      un 403 cuando la petición no declara un User-Agent válido.

    Si todos los proveedores responden correctamente pero el mapa interactivo
    sigue sin fondo dentro del notebook, la causa es el visor del editor y no
    la red: basta con abrir el archivo HTML en el navegador.

    Parameters
    ----------
    timeout : float, optional
        Segundos de espera por proveedor.

    Returns
    -------
    pandas.DataFrame
        Una fila por proveedor con ``estado``, ``codigo_http``, ``bytes`` y
        ``detalle``.
    """
    import urllib.error
    import urllib.request

    filas = []
    z, x, y = _MOSAICO_PRUEBA["z"], _MOSAICO_PRUEBA["x"], _MOSAICO_PRUEBA["y"]
    for clave, cfg in MOSAICOS.items():
        url = cfg["url"].format(z=z, x=x, y=y)
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        fila = {"proveedor": cfg["nombre"], "clave": clave}
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                datos = resp.read()
                fila.update(estado="OK", codigo_http=resp.status,
                            bytes=len(datos), detalle="")
        except urllib.error.HTTPError as exc:
            fila.update(estado="RECHAZADO", codigo_http=exc.code, bytes=0,
                        detalle=str(exc.reason)[:80])
        except urllib.error.URLError as exc:
            fila.update(estado="SIN CONEXIÓN", codigo_http=None, bytes=0,
                        detalle=str(exc.reason)[:80])
        except Exception as exc:  # noqa: BLE001
            fila.update(estado="ERROR", codigo_http=None, bytes=0,
                        detalle=f"{type(exc).__name__}: {str(exc)[:60]}")
        filas.append(fila)

    return pd.DataFrame(filas)


def abrir_en_navegador(ruta_html) -> bool:
    """Abre un mapa HTML guardado en el navegador predeterminado.

    Necesario en VS Code: su visor de notebooks restringe las descargas
    externas dentro de las salidas de las celdas, por lo que los mapas
    interactivos aparecen con fondo gris aunque la red funcione. Abiertos
    en el navegador se ven completos.

    Parameters
    ----------
    ruta_html : str | pathlib.Path

    Returns
    -------
    bool
        ``True`` si el sistema pudo lanzar el navegador.
    """
    import webbrowser
    from pathlib import Path

    ruta = Path(ruta_html).resolve()
    if not ruta.exists():
        log.warning("No existe el archivo %s", ruta)
        return False
    abierto = webbrowser.open(ruta.as_uri())
    log.info("Mapa %s en el navegador: %s", "abierto" if abierto else "no abierto",
             ruta.name)
    return abierto
