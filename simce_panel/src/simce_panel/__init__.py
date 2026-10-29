"""
simce_panel
===========

Pipeline reproducible para construir el panel longitudinal SIMCE 2° medio
(2015-2025) y georreferenciarlo con el Directorio Oficial de Establecimientos.

Cubre el flujo completo de la tesis: construcción y saneamiento del panel
(Capítulo 3), variables de resultado, dataset analítico y análisis espacial.

Submódulos
----------
Construcción del panel

- :mod:`config`      : rutas, niveles y alias de columnas.
- :mod:`columnas`    : resolución robusta de nombres de columnas.
- :mod:`io_datos`    : lectura/escritura de CSV/Excel y utilidades.
- :mod:`unificacion` : apilado longitudinal por nivel.
- :mod:`directorio`  : cruce con el Directorio y saneamiento de coordenadas.
- :mod:`validacion`  : reportes y chequeos de integridad.

Armonización y calidad

- :mod:`glosarios`       : catálogos de códigos por año, leídos de los xlsx.
- :mod:`equivalencias`   : alias de texto observados y categorías canónicas.
- :mod:`estandarizacion` : traducción de códigos a texto, por año.
- :mod:`diagnostico`     : supresiones, cobertura y saneamiento de puntajes.
- :mod:`panel_ancho`     : pivoteo a una fila por establecimiento.
- :mod:`glosario`, :mod:`glosario_unificado` : diccionarios de datos.

Análisis

- :mod:`resultado` : caída inicial, recuperación y recuperación relativa.
- :mod:`dataset`   : dataset analítico y variables geográficas derivadas.
- :mod:`espacial`  : pesos espaciales, índice de Moran y LISA con corrección FDR.
- :mod:`mapas`     : mapas interactivos y estáticos sobre cartografía real.
- :mod:`bayes`     : regresión bayesiana jerárquica de serie interrumpida.
- :mod:`robustez`  : verificaciones de la revisión del 1-oct-2026.
- :mod:`covariables` : matrícula, participación, IVE, pobreza, densidad, distancia.
- :mod:`bym2`      : efecto espacial BYM2 por comuna.
- :mod:`contrastes`: H5 emparejada y gradiente SHAP con bootstrap.
- :mod:`prediccion`: validación predictiva con 2025 reservado (criterio C3).

Uso rápido
----------
>>> from simce_panel import Config, unificar_todos, enriquecer_rbd
>>> cfg = Config.desde_base("/ruta/al/proyecto")
>>> paneles = unificar_todos(cfg)                 # crea los 4 unificados
>>> rbd = enriquecer_rbd(paneles["rbd"], cfg)     # añade lat/long al RBD
"""

from __future__ import annotations

import logging

from . import (bayes, bym2, columnas, config, contrastes, covariables, dataset, diagnostico, directorio,
               equivalencias, espacial, estandarizacion, glosario,
               glosario_unificado, glosarios, io_datos, mapas, panel_ancho, prediccion,
               resultado, robustez, unificacion, validacion)
from .config import Config
from .directorio import enriquecer_rbd
from .glosario import exportar_glosario_excel
from .unificacion import unificar_nivel, unificar_todos
from .validacion import reporte_unificacion, resumen_por_anio

__all__ = [
    "Config",
    "config",
    "columnas",
    "io_datos",
    "unificacion",
    "directorio",
    "validacion",
    "glosario",
    "diagnostico",
    "glosarios",
    "estandarizacion",
    "panel_ancho",
    "glosario_unificado",
    "resultado",
    "dataset",
    "espacial",
    "mapas",
    "bayes",
    "robustez",
    "bym2",
    "covariables",
    "contrastes",
    "prediccion",
    "equivalencias",
    "unificar_nivel",
    "unificar_todos",
    "enriquecer_rbd",
    "exportar_glosario_excel",
    "reporte_unificacion",
    "resumen_por_anio",
    "configurar_logging",
]

__version__ = "1.1.2"


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura un logging simple que imprime mensajes en el notebook.

    Parameters
    ----------
    nivel : int, optional
        Nivel de logging (por defecto ``logging.INFO``).
    """
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
