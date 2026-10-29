# Datos del proyecto

Todas las fuentes son públicas y se publican agregadas por establecimiento
(RBD) o por comuna; ninguna contiene datos personales. Los archivos originales
no se versionan por su tamaño: se descargan de los portales oficiales y se
dejan en las carpetas indicadas. Las carpetas `Unificados/` y `muestra/` las
llenan los cuadernos y `herramientas/generar_muestra.py`.

## Fuentes y dónde dejarlas

| Fuente | Institución | Años | Portal de descarga | Carpeta |
|---|---|---|---|---|
| Resultados SIMCE 2.º medio por establecimiento (y sus glosas) | Agencia de Calidad de la Educación | 2015–2018, 2022–2025 | informacionestadistica.agenciaeducacion.cl | `Simce 2°medio/<año>/` |
| Directorio Oficial de Establecimientos | Ministerio de Educación, Centro de Estudios | 2014–2025 | datosabiertos.mineduc.cl | `Directorio 2014-2025/` |
| Resumen de Matrícula por establecimiento | Ministerio de Educación, Centro de Estudios | 2015–2018, 2022–2025 | datosabiertos.mineduc.cl | `Externos/` |
| Resumen de Matrícula por curso | Ministerio de Educación, Centro de Estudios | 2015–2018, 2022–2025 | datosabiertos.mineduc.cl | `Externos/` |
| IVE-SINAE e IVM por establecimiento | JUNAEB | 2015–2025 | www.junaeb.cl | `Externos/` |
| Pobreza por ingresos comunal (estimación de áreas pequeñas, SAE) | Ministerio de Desarrollo Social y Familia, sobre la CASEN | 2015 y 2017 | observatorio.ministeriodesarrollosocial.gob.cl | `Externos/` |
| Superficie comunal | Biblioteca del Congreso Nacional | vigente | www.bcn.cl | `Externos/` |

Notas:

- La base SIMCE 2025 es la versión **preliminar** que la Agencia liberó en
  abril de 2026. Si se publica la definitiva, conviene reemplazarla y volver a
  ejecutar los cuadernos.
- Los nombres de archivo cambian entre años; el paquete los reconoce por
  patrones (`simce_panel.config.ALIAS_COLUMNAS` y `simce_panel.io_datos`), de
  modo que basta con dejar cada archivo en la carpeta de su año.
- Las bases SIMCE traen una columna `marca_<materia>2m_rbd`. El cuaderno 02
  deja vacíos los puntajes marcados como no representativos ("ajenas a la
  Agencia", "ajenas al establecimiento", "alumnos insuficientes") y los códigos
  de supresión (puntaje menor que 100).

## Archivos de `muestra/`

Contienen el 15 % de los establecimientos (475, estratificado por región,
semilla 2026). El nombre del establecimiento se reemplazó por un rótulo
genérico; el RBD y las coordenadas se conservan porque los cuadernos los
necesitan y ya son públicos en el Directorio Oficial.

### `simce_rbd_ancho_reducido_2015_2025.csv`

Una fila por establecimiento. Separador `;`, codificación UTF-8 con BOM.

| Variable | Descripción |
|---|---|
| `rbd` | Rol Base de Datos: identificador oficial del establecimiento |
| `nom_rbd` | Nombre (en la muestra, rótulo genérico) |
| `latitud`, `longitud` | Coordenadas del Directorio Oficial (WGS84), última disponible |
| `region_dir`, `comuna_dir` | Región y comuna según el Directorio Oficial |
| `cod_depe1_cat` | Dependencia detallada (municipal DAEM, corporación, SLEP, particular subvencionado, particular pagado, administración delegada), último año |
| `cod_depe2_cat` | Dependencia agrupada (municipal, SLEP, particular subvencionado, particular pagado), último año |
| `cod_grupo_cat` | Grupo socioeconómico de la Agencia (Bajo a Alto), último año |
| `cod_rural_rbd_cat` | Urbano o rural |
| `cod_depe1_cambio`, `cod_grupo_cambio` | Verdadero si la dependencia o el GSE cambiaron entre años |
| `n_anios_presente` | Años con resultados SIMCE (máximo 8) |
| `prom_<materia>2m_rbd_<año>` | Puntaje promedio del establecimiento (`mate` o `lect`); vacío si no rindió o el resultado no es representativo |
| `nalu_<materia>2m_rbd_<año>` | Estudiantes evaluados |

### `simce_rbd_unificado_2015_2025.csv`

Panel largo: una fila por establecimiento y año, solo con las columnas que
usan los cuadernos 04, 06 y 09. Los puntajes ya vienen saneados.

| Variable | Descripción |
|---|---|
| `rbd`, `año` | Establecimiento y año de aplicación |
| `nom_rbd` | Nombre (en la muestra, rótulo genérico) |
| `cod_depe1`, `cod_depe2` | Dependencia, tal como la publica la Agencia (texto hasta 2017, código desde 2018) |
| `cod_grupo` | Grupo socioeconómico del año, con la misma codificación de origen |
| `cod_rural_rbd` | Ruralidad |
| `prom_<materia>2m_rbd`, `nalu_<materia>2m_rbd` | Puntaje promedio y evaluados del año |
| `marca_<materia>2m_rbd` | Marca de la Agencia sobre el resultado (p. ej., no representativo) |
| `latitud`, `longitud` | Coordenadas del Directorio Oficial |

### `resultados_mate_2m.csv` y `resultados_lect_2m.csv`

Una fila por establecimiento; las salidas del cuaderno 02 para la muestra.

| Variable | Descripción |
|---|---|
| `puntaje_pre`, `n_pre` | Promedio 2015–2018 y años válidos en ese período |
| `puntaje_impacto` | Puntaje 2022 |
| `caida_inicial` | 2022 menos el promedio previo (negativo = cayó) |
| `recuperacion`, `n_post` | Pendiente de la recta 2022–2025, en puntos por año, y años válidos |
| `recuperacion_relativa` | Recuperado hasta el último año ÷ magnitud de la caída; solo si cayó al menos 1 punto |
| `puntaje_post_medio`, `puntaje_post_prom` | Promedio del período 2022–2025 |
| `puntaje_post_ultimo`, `anio_post_ultimo` | Último puntaje disponible y su año |
| `brecha_simple` | Promedio posterior menos promedio previo; solo como referencia frente a los reportes oficiales |
| `muestra_analitica` | Verdadero si la caída y la recuperación son calculables (al menos 2 años previos y 3 posteriores) |
| `puntaje_tardio`, `desvio_tardio`, `tipo_trayectoria`, `ruptura_sostenida` | Clasificación de la trayectoria (normal, atípico 2022, ruptura sostenida, sin dato); solo en Matemática |
