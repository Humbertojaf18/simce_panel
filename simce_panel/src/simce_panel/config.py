"""
config.py
=========

Configuración central del pipeline de construcción del panel SIMCE.

Aquí se concentran todas las constantes y rutas del proyecto para que el
resto de los módulos no contengan rutas "hardcodeadas". De este modo, mover
el dataset a otra máquina solo requiere cambiar la ruta base en un único
lugar (o pasarla por argumento).

Contenido principal
-------------------
- :class:`Config`        : dataclass con todas las rutas derivadas de una base.
- :data:`NIVELES`        : niveles de agregación del SIMCE.
- :data:`ALIAS_COLUMNAS` : diccionario de alias para resolver nombres de
                           columnas que cambian entre años (rbd, lat, long...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constantes de dominio
# --------------------------------------------------------------------------- #

#: Niveles de agregación que entrega la Agencia de Calidad de la Educación.
#: Para cada uno existe un CSV por año que se apilará en un único archivo.
NIVELES: tuple[str, ...] = ("comuna", "deprov", "region", "rbd")

#: Nombre estándar de la columna de año que añade el pipeline a cada panel.
COL_ANIO: str = "año"

#: Nombres estándar de las columnas de coordenadas tras el cruce con Directorio.
COL_LAT: str = "latitud"
COL_LON: str = "longitud"
COL_RBD: str = "rbd"

#: Alias conocidos por concepto. Se comparan de forma "normalizada"
#: (sin acentos, en minúsculas y sin separadores), por lo que basta con
#: listar las variantes razonables que aparecen en los archivos del MINEDUC
#: y de la Agencia de Calidad.
ALIAS_COLUMNAS: dict[str, list[str]] = {
    "rbd": ["rbd", "cod_rbd", "rbd_ee", "id_rbd"],
    "anio": ["año", "agno", "anio", "ano", "year"],
    "latitud": ["latitud", "gps_lat", "lat", "latitude", "y", "latitud_ee"],
    "longitud": ["longitud", "gps_long", "lng", "lon", "long", "longitude",
                 "x", "longitud_ee"],
    # Variables administrativas útiles aguas abajo (Capítulo 3 de la tesis:
    # dependencia, ruralidad, región). Se dejan declaradas para reutilizarlas
    # al enriquecer el panel RBD si más adelante se requieren.
    "dependencia": ["cod_depe", "cod_depe2", "dependencia", "depe"],
    "ruralidad": ["rural_rbd", "ruralidad", "rural", "cod_rural"],
    "region": ["cod_reg_rbd", "region", "nom_reg_rbd_a", "cod_region"],
    "comuna": ["cod_com_rbd", "comuna", "nom_com_rbd", "cod_comuna"],
}


# --------------------------------------------------------------------------- #
# Configuración de rutas
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """Rutas y parámetros de E/S del proyecto.

    La forma recomendada de crear una instancia es :meth:`desde_base`,
    pasándole la carpeta que contiene ``Dataset``.

    Attributes
    ----------
    dataset_dir : Path
        Carpeta ``Dataset`` con los datos estandarizados.
    nombre_dir_directorio : str
        Nombre de la subcarpeta con los Directorios oficiales por año.
    nombre_dir_simce : str
        Nombre de la subcarpeta con los SIMCE 2° medio por año.
    nombre_dir_salida : str
        Subcarpeta (dentro de ``Dataset``) donde se guardan los unificados.
    sep : str
        Separador de los CSV (por defecto ``;``).
    encoding : str
        Codificación de los CSV (por defecto ``utf-8-sig``).
    """

    dataset_dir: Path
    nombre_dir_directorio: str = "Directorio 2014-2025"
    nombre_dir_simce: str = "Simce 2°medio"
    nombre_dir_salida: str = "Unificados"
    sep: str = ";"
    encoding: str = "utf-8-sig"

    # Campos derivados (se calculan en __post_init__).
    directorio_dir: Path = field(init=False)
    simce_dir: Path = field(init=False)
    salida_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.directorio_dir = self.dataset_dir / self.nombre_dir_directorio
        self.simce_dir = self.dataset_dir / self.nombre_dir_simce
        self.salida_dir = self.dataset_dir / self.nombre_dir_salida

    @classmethod
    def desde_base(cls, base_dir: str | Path, **kwargs) -> "Config":
        """Crea la configuración a partir de la carpeta que contiene ``Dataset``.

        Parameters
        ----------
        base_dir : str | Path
            Carpeta que contiene la carpeta ``Dataset`` (p. ej. la raíz del
            proyecto o de Google Drive).
        **kwargs
            Cualquier atributo de :class:`Config` que se quiera sobreescribir
            (por ejemplo ``nombre_dir_simce`` si la carpeta tiene otro nombre).

        Returns
        -------
        Config
        """
        return cls(dataset_dir=Path(base_dir) / "Dataset", **kwargs)

    def ruta_salida(self, nombre_archivo: str) -> Path:
        """Devuelve la ruta completa de un archivo dentro de ``Unificados``."""
        return self.salida_dir / nombre_archivo

    def crear_salida(self) -> Path:
        """Crea (si no existe) la carpeta de salida y la devuelve."""
        self.salida_dir.mkdir(parents=True, exist_ok=True)
        return self.salida_dir

    def validar(self) -> None:
        """Verifica que existan las carpetas de entrada esperadas.

        Raises
        ------
        FileNotFoundError
            Si falta ``Dataset``, la carpeta de Directorio o la de SIMCE.
        """
        faltantes = [
            p for p in (self.dataset_dir, self.directorio_dir, self.simce_dir)
            if not p.exists()
        ]
        if faltantes:
            detalle = "\n".join(f"  - {p}" for p in faltantes)
            raise FileNotFoundError(
                "No se encontraron las siguientes carpetas esperadas:\n"
                f"{detalle}\n"
                "Revisa el parámetro 'base_dir' o los nombres de subcarpeta."
            )


#: Plantillas de nombre para los archivos unificados de salida.
NOMBRES_SALIDA: dict[str, str] = {
    "comuna": "simce_comuna_unificado_2015_2025.csv",
    "deprov": "simce_deprov_unificado_2015_2025.csv",
    "region": "simce_region_unificado_2015_2025.csv",
    "rbd": "simce_rbd_unificado_2015_2025.csv",
}

#: Nombre del archivo de glosario unificado.
NOMBRE_GLOSARIO: str = "glosario_unificado_2015_2025.xlsx"
