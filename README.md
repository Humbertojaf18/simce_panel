# Caída y recuperación del rendimiento SIMCE post-pandemia (2015–2025)

Código, cuadernos y descripción de datos del proyecto de tesis
*Caída y recuperación del rendimiento SIMCE post-pandemia en establecimientos
chilenos (2015–2025): trayectorias, composición socioeconómica y territorio*.

- **Autor:** Humberto Arrieta
- **Profesor guía:** Dante Travisany Flores · **Coguía (Capstone Project):** Christian Pieringer Baeza
- **Programa:** Magíster en Ciencia de Datos, Universidad de Las Américas · Capstone I, octubre de 2026

El proyecto mide, para cada establecimiento con 2.º medio, cuánto cayó su
puntaje SIMCE en 2022 respecto de su trayectoria previa (2015–2018) y cuánto
se recuperó entre 2022 y 2025; estima esas trayectorias con un modelo
bayesiano jerárquico de serie interrumpida y contrasta cinco hipótesis sobre
composición socioeconómica, territorio, dependencia y traspaso a los SLEP.

## Estructura

```
.
├── 01_construccion_panel.ipynb … 12_contraste_h4.ipynb   cuadernos, en orden de ejecución
├── simce_panel/
│   ├── CAMBIOS.md                  historial de versiones del paquete
│   └── src/simce_panel/            paquete de Python (un módulo por etapa)
├── Dataset/
│   ├── README.md                   fuentes, descarga y diccionario de datos
│   ├── muestra/                    muestra pública para verificar la ejecución
│   ├── Directorio 2014-2025/       (no versionado) Directorio Oficial, un archivo por año
│   ├── Simce 2°medio/              (no versionado) bases SIMCE por establecimiento, una carpeta por año
│   ├── Externos/                   (no versionado) matrícula, IVE, pobreza comunal, superficie
│   └── Unificados/                 (no versionado) salidas intermedias que generan los cuadernos
├── herramientas/                  generar_muestra.py y limpiar_salidas_identificables.py
├── requirements.txt              bibliotecas sin versión fija
├── requirements-lock.txt         versiones exactas de un entorno verificado con la muestra
├── .gitignore
└── README.md
```

Los cuadernos buscan el paquete en `simce_panel/src` y los datos en
`Dataset/`, ambos junto a los cuadernos (`BASE_DIR = "."`).

## Instalación

Probado con **Python 3.12.13** en Windows 11; la muestra se probó además con Python 3.13 en Linux.

```bash
python -m venv .venv
.venv\Scripts\activate            # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lista las bibliotecas sin fijar versiones. Para instalar
exactamente las versiones con que se verificó la ejecución de la muestra
(cuadernos 04, 05, 06, 08, 09 y 10, Python 3.12):

```bash
pip install -r requirements-lock.txt
```

Ese entorno se comprobó con la muestra pública; los cuadernos publicados se
ejecutaron con Python 3.12.13 en Windows 11, y las versiones exactas de ese
entorno no quedaron registradas.

## Datos

Todas las fuentes son públicas, están agregadas por establecimiento o por
comuna y no contienen datos personales. Los archivos originales no se
versionan por su tamaño; `Dataset/README.md` indica de dónde descargar cada
uno, en qué carpeta dejarlo y qué contiene cada variable.

## Verificación rápida con la muestra

`Dataset/muestra/` contiene el 15 % de los establecimientos (475, estratificado
por región, semilla 2026): el panel ancho reducido, las columnas del panel
largo que usan los cuadernos y las variables de resultado de Matemática y
Lectura, con el nombre del establecimiento reemplazado por un rótulo genérico. Permite ejecutar los cuadernos 04, 05, 06,
08, 09 y 10 sin descargar nada: basta agregar `nombre_dir_salida="muestra"` a
la línea de configuración de cada cuaderno, por ejemplo

```python
cfg = sp.Config.desde_base(".", nombre_dir_salida="muestra")
```

Los cuadernos escriben sus salidas en la misma carpeta `muestra/`.

Con la muestra los cuadernos corren de punta a punta, pero las cifras no son
las del informe: los contrastes espaciales y los intervalos dependen del
tamaño de la muestra. Los cuadernos 01–03 (construcción del panel) y 07, 11 y
12 (covariables externas) requieren los datos completos.

## Reproducción completa

Ejecutar los cuadernos en orden. Cada uno lee las salidas de los anteriores
desde `Dataset/Unificados/`.

| Cuaderno | Qué hace | Salidas principales | Informe |
|---|---|---|---|
| 01 construcción del panel | Unifica las bases SIMCE y el Directorio; georreferencia | `simce_*_unificado_2015_2025.csv` | 3.2 |
| 02 revisión de calidad | Sanea puntajes y construye las variables de resultado | `resultados_{mate,lect}_2m.csv` | 3.2–3.3 |
| 03 base unificada | Pivota a una fila por establecimiento | `simce_rbd_ancho_reducido_2015_2025.csv` | 3.2 |
| 04 dataset analítico y EDA | Variables geográficas, sesgo de selección | `dataset_analitico_{mate,lect}.csv` | 3.2.3, 4.1 |
| 05 análisis espacial | Moran y LISA sobre residuos (H1, H2) | `espacial_mate.csv`, mapas | 4.2 |
| 06 modelo bayesiano | Serie interrumpida jerárquica por escuela | `bayes_escuelas_mate.csv`, `bayes_grupos_mate.csv` | 4.3.1 |
| 07 covariables adicionales | Vulnerabilidad, pobreza, densidad, participación | `dataset_analitico_mate_covariables.csv` | 3.2, 4.3.3 |
| 08 BYM2 comunal | Estructura espacial a escala de comuna | tablas del cuaderno | 4.2 |
| 09 contrastes H3–H5 | Gradiente socioeconómico y traspaso a los SLEP | tablas del cuaderno | 4.3, 4.4.1 |
| 10 validación predictiva | Año 2025 reservado; M0–M4 (criterio C3) | `validacion_predictiva_2025.csv` | 4.5 |
| 11 modelo con covariables | Robustez de H3 | `h3_robustez_covariables.csv` | 4.3.2 |
| 12 contraste H4 | Ruralidad y dependencia | `h4_contraste.csv` | 4.4.2 |

Los procesos aleatorios usan la semilla 2026 (MCMC, permutaciones de Moran y
LISA, bootstrap), de modo que una segunda ejecución reproduce las cifras salvo
diferencias en el último decimal debidas al hardware. Los modelos bayesianos
corren en CPU; en un equipo convencional, el cuaderno 06 tarda unos minutos y
el 08, unos quince.

## Licencia

Por definir por el autor (código). Los datos pertenecen a sus instituciones de
origen y se rigen por sus condiciones de uso.
