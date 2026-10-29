# simce_panel 1.1.2 — cambios de la auditoría del 24-oct-2026

## 1.1.2 (revisión de datos del 24-oct, tarde)

Cambian `diagnostico.py`, `__init__.py` y el cuaderno 03.

| Qué | Cambio | Por qué | Cuaderno |
|---|---|---|---|
| Cuaderno 03 | Aplica `sanear_puntajes` antes de pivotar (sección 4) | El cuaderno 02 sanea el panel solo en memoria para construir las medidas; el panel ancho que usan los cuadernos 04 a 12 conservaba 153 puntajes de Matemática y 143 de Lectura de 2016–2017 marcados por la Agencia como no representativos, y dos ceros de 2022 por materia | 03, y luego 04 a 12 |
| `diagnostico.py` | `sanear_puntajes` usa la columna exacta `marca_<m>2m_rbd` cuando existe | En un panel estandarizado conviven la marca original y sus capas `_desc` y `_cat` | 02, 03 |
| `__init__.py` | `__version__ = "1.1.2"` | | |

**Efecto medido (réplica con tus datos, antes de que re-ejecutes).** Sanear el panel ancho cambia las caídas y recuperaciones por grupo del modelo bayesiano en 0,1 puntos o menos; el gradiente de H3 en 0,02 (GSE reciente: Matemática 0,09 → 0,08, Lectura 1,53 → 1,52; GSE previo: Matemática −1,37 → −1,39, Lectura 1,17 → 1,18); y la validación predictiva en 0,03 puntos de RMSE (M3 16,93 → 16,91; M3b 15,66; M4 15,65 → 15,64; cobertura 0,945 → 0,947). Los dos ceros ya los descartaba el filtro de 100 puntos del modelo y de la validación. Las cifras del informe se mantienen; re-ejecutar 03 → 12 queda para el inicio de Capstone II, y el panel ancho saneado reproduce exactamente las 2.821 escuelas del cuaderno 02.

## 1.1.1 (después de tu ejecución con la 1.1.0)

Solo cambian tres archivos: copia `bym2.py`, `robustez.py` y `__init__.py` encima de los de la 1.1.0.

| Módulo | Cambio | Por qué | Cuaderno |
|---|---|---|---|
| `bym2.py` | Cadenas de 2.000 + 3.000 iteraciones por defecto (antes 1.500 + 1.500) | En tu ejecución el R-hat de ρ quedó en 1,011 y 1,012, sobre el umbral de 1,01, con tamaño efectivo de 390 y 241 | 08 |
| `robustez.py` | Silencia el aviso «divide by zero» de scipy y corrige el comentario sobre coordenadas repetidas | El aviso es inocuo; las coordenadas repetidas no daban peso infinito, sino que esos pares no contaban como vecinos | 05 (no hace falta reejecutar) |
| `__init__.py` | `__version__ = "1.1.1"` | | |

## 1.1.0

Ocho módulos cambian. El resto del paquete queda igual que en la 1.0.0.

| Módulo | Cambio | Por qué | Cuaderno |
|---|---|---|---|
| `espacial.py` | `modelo_base` ajusta con Powell (antes L-BFGS) y verifica que la log-verosimilitud sea finita; avisa si los residuos no tienen media cercana a 0 | L-BFGS se detenía en un borde degenerado: log-verosimilitud infinita, intercepto 0, varianza comunal 0 y residuos con media 103 | 05 |
| `espacial.py` | `moran_global` y `lisa_con_fdr` aceptan `semilla` (por defecto 2026) | Sin semilla, valores p y conteos LISA cambiaban en cada ejecución | 05 |
| `robustez.py` | `moran_con_contraccion` centra el residuo antes de contraer | Contraer un residuo sin centrar agregaba variación ligada al tamaño de la escuela (0,063 en vez de 0,036) | 05 |
| `robustez.py` | `sensibilidad_moran` separa 50 m las coordenadas repetidas, fija la semilla y silencia el aviso «divide by zero» de scipy | Tres escuelas de la muestra comparten coordenadas: con distancia inversa no se contaban como vecinas. El efecto en el Moran es de una diezmilésima; el aviso es inocuo | 05 |
| `resultado.py` | `recuperacion_relativa` solo para escuelas con caída ≤ −1 punto | Antes incluía a 865 escuelas que subieron en 2022, para las que el cociente no mide recuperación | 02 |
| `contrastes.py` | `h5_emparejado` calcula el MDE como 2,8 × error estándar bootstrap del ATT y lo devuelve (`EE_bootstrap`) | La fórmula de dos muestras independientes no corresponde a un ATT de pares y subestimaba el MDE (0,64 en vez de 0,77) | 09 |
| `prediccion.py` | `prediccion_mixto(..., caida_aleatoria=True)` ajusta M3b, con la caída de 2022 aleatoria por escuela | Línea base justa para C3: M3 obliga a todas las escuelas a caer igual | 10 |
| `bayes.py` | Nueva `diagnosticos_escuela(mcmc)`: R-hat y ESS de la caída y la recuperación de cada escuela | C2 se verificaba solo en parámetros de grupo | 06 |
| `bayes.py` | Nueva `gse_modal_previo(largo)`: GSE modal 2015–2018 (empates: el del año más reciente) | El GSE del año más reciente es posterior al tratamiento; 25 % de las escuelas cambió de grupo | 06, 09 |
| `bym2.py` | Factor de escala por componente conexa, con la inversa generalizada exacta del laplaciano; restricción de suma cero por componente; ICAR con `ImproperUniform`; comunas aisladas solo con parte independiente; 4 cadenas de 1.500 + 1.500 por defecto; devuelve `r_hat_rho` y `ess_rho` | El factor salía 8.962 en vez de 0,78 y anulaba la parte espacial: ρ no podía estimarse | 08 |
| `__init__.py` | `__version__ = "1.1.0"` | Para verificar qué versión corre cada cuaderno | todos |

## Salidas esperadas principales

- 02/04: recuperación relativa con n = 1.856, mediana 0,60.
- 05: Moran del residuo 0,0316 (caída) y 0,0161 (recuperación); varianza comunal 6,02 y 0,57; contraído 0,0364; atenuación 0,0397.
- 06: R-hat por escuela ≤ 1,004; ESS por escuela ≥ 1.900; con GSE previo (719 escuelas cambian de grupo), GSE Bajo −9,8 frente a su tendencia.
- 08: fracción comunal 0,045 [0,022; 0,086] y ρ 0,84 [0,25; 1,00] en la caída; 0,025 y 0,68 [0,09; 1,00] en la recuperación.
- 09: ATT 0,048 [−0,473; 0,604], MDE ≈ 0,77; H3 con GSE previo: Matemática −1,37 [−2,58; −0,16], Lectura +1,19 [0,26; 2,05].
- 10: M3 16,93; M3b 15,66; M4 15,65.
