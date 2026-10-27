# Auditoría y stress test de `norm="sv"` — 27 octubre 2026

## Fuente y contrato del algoritmo

[Highly Adaptive Empirical Risk Minimization with Principal Components, v1, sección 5](https://arxiv.org/html/2603.18204v1#S5) propone una actualización multiplicativa de alpha con dirección proyectada sobre el tangente a `||V alpha||1 = C`. La inicialización ridge o lasso fija `C(lambda)=||V alpha0(lambda)||1`; lambda no es directamente un radio SV. La implementación pública mantiene esa forma de elegir el radio y la igualdad, sin añadir un argumento C nuevo.

En una cara suave, el representador de la derivada es `D=alpha*grad R`; descenso usa su negativo. El paper escribe el camino con `+D*` después de definir D como derivada, aunque pide disminuir el riesgo: hay que revisar ese signo en el manuscrito. El código ya usaba el signo de descenso correcto; no era un bug que hubiese que invertir.

La definición teórica por bola `||V alpha||1<=C` y el algoritmo sobre la frontera `=C` deben distinguirse. La igualdad no define un conjunto convexo; estacionariedad multiplicativa no certifica un mínimo global y, con ceros, ni siquiera un mínimo local en toda la frontera.

## Errores reproducidos y reparación

1. Se omitían beta_j nulos o muy pequeños al dividir por beta_j para limitar el paso. Si se activa un beta_j nulo, la ortogonalidad calculada con `sign(0)=0` no conserva L1. Se añade una retracción radial exacta al radio inicial antes de aceptar cualquier paso. Para una cara suave con paso factible, coincide con el camino del paper salvo redondeo; en los puntos no diferenciables es una extensión explícita, no una fórmula atribuida al artículo.
2. No había búsqueda que garantizase descenso: se aceptaban riesgos mayores e incluso candidatos no finitos. Ahora Armijo comprueba la disminución con el desplazamiento realmente retraído y rechaza candidatos no finitos. Si no hay paso aceptable, se devuelve el último iterado válido y `line_search_stalled`.
3. Se escogía uno de los últimos 100 candidatos por el gradiente del punto anterior. Ese valor no diagnosticaba el candidato devuelto y podía seleccionar riesgo mayor. Ahora se devuelve el último paso aceptado y se recalcula su score tangente.
4. La proyección usaba un corte absoluto sobre la norma al cuadrado del normal, incorrecto para radios pequeños. Ahora se normaliza el normal antes de proyectar.
5. El ajuste gaussiano Python y su CV recomputaban los vectores propios independientemente de V. Se reutilizan U,d,V de una sola descomposición: elimina trabajo y evita incompatibilidades de signo/rotación. El wrapper single ya no recomputa otra descomposición para reconstruir beta y expone el historial real.
6. El intercepto logístico se resolvía mediante Newton sin salvaguarda. Se comparte un cálculo acotado entre el riesgo SV y el refit; Python usa la misma regla. Folds de una sola clase tienen un límite numérico finito explícito, pues su intercepto MLE no es finito.

Nuevos diagnósticos low-level: `status`, `sv_radius`, `tangent_norm`; en R son atributos `sv_status`, `sv_radius`, `sv_tangent_norm`. El single gaussiano los conserva. Aún no se propagan los estados de todos los folds/refits a todas las salidas de CV/binomial; no confundir una salida de esas interfaces con un certificado de convergencia.

## Stress geométrico de las tres normas

Datos simulados: indicadores HAL univariantes en n−1 nudos interiores distintos, centrados, 12 PCs, n=32,64,128,256. Se calibra el camino nativo por bisección para que cada norma tenga radio numérico C=1 **en su propio espacio**. Los modos 1/2 se ajustan con sus fórmulas nativas; SV comienza en ridge de radio inducido 1 y ejecuta el solver nativo revisado. No se elige una lambda común: tiene distinta interpretación en cada modo. No es una comparación por CV ni de eficacia predictiva a igual complejidad.

| n | SV de coeficientes con norm=1 | con norm=2 | con norm=sv |
|---:|---:|---:|---:|
| 32 | 5.0889 | 5.0923 | 1.0000 |
| 64 | 7.2011 | 7.1948 | 1.0000 |
| 128 | 10.1854 | 10.1764 | 1.0000 |
| 256 | 14.4049 | 14.4185 | 1.0000 |

Máximo error absoluto del radio en el historial SV: 7.8e−16. Tres ajustes SV terminan `line_search_stalled` tras 138, 212 y 1026 pasos; el cuarto alcanza `max_iter` (3000): **hay factibilidad, no un certificado de ERM**. CSV conserva MSE y estado: SV tiene mayor error de entrenamiento, como cabe esperar con una clase mucho más restringida. La señal de este stress depende de n y no tiene SV fija; el experimento demuestra geometría del control, no una tasa estadística ni superioridad predictiva.

La cantidad controlada es `||V alpha||1`, parte de variación de los coeficientes. Para átomos canónicos distintos es la parte no constante de la norma SV. El intercepto anclado se reconstruye al descentrar: `a0=ymean-means(H)' beta`; la norma completa es `|a0|+||beta||1`. El CSV la incluye por separado (aproximadamente 1.48–1.49 en SV, no 1). En el modo logístico la función cuya variación se considera es el predictor logit. No se afirma control de la norma completa por C si el intercepto queda libre.

## Regresiones y comparación con la versión previa

En 40 casos con V ortonormal y beta_1 nulo por cancelación (20 gaussianos, 20 binomiales), el código de la revisión base pierde el radio en los 40. Deriva relativa máxima del historial: 1.525 gaussiana y 2.058 binomial. La revisión conserva el radio a precisión numérica. En gaussianos el máximo aumento de MSE aceptado por la base fue 2.90e6; en la revisión no hay aumentos por encima del error de redondeo entre evaluaciones Eigen/NumPy (máximo 1.82e−12 sobre riesgos de gran magnitud). Los 40 casos revisados dan pasos reales; se identifican los ceros numéricos con una tolerancia proporcional a C, y la retracción mantiene la norma incluso si se activan. Son diseños de estrés, no datos HAL ni una campaña de rendimiento.

- 9 tests Python pasan: 40 trayectorias aleatorias, cancelación exacta, solución conocida en una cara suave, radio pequeño, radio cero, sin pasos, validación, historial/predicciones públicas e interceptos extremos.
- Pruebas R propias pasan: gaussianos nativos/públicos con ambas inicializaciones, logístico nativo/público, predicciones, norma, descenso y controles.
- Suite R binomial existente: 77 comprobaciones pasan, 0 fallos. 200 warnings de locale (`LC_ALL=C.UTF-8`) emitidos por withr/testthat, registrados aparte.
- Compilaciones R y Python nuevas en directorios temporales, sin sustituir instalaciones/binaries del usuario. No se ha ejecutado la colección Python histórica completa, que no está disponible en este checkout.

## Reproducción

Con el paquete **recompilado** desde estas fuentes, numpy y scipy disponibles:

```bash
python tests/test_sv_descent.py -v
Rscript tests/r/test_sv_descent.R
python examples/sv_stress/stress.py --output /tmp/hapc-sv-results
python examples/sv_stress/baseline_compare.py --output /tmp/hapc-sv-baseline
```

La última orden compila ambos kernels en un directorio temporal; requiere clang++ y Eigen (la ruta se puede indicar con `--eigen-include`). Baseline explícita: commit `84bdb8ac54f0858965c29d6bb9d17c571500fe8a`. Usa solo esos tres archivos históricos, conservando los demás cambios actuales. [Resultados y logs](results/2026-10-27/) y [diseño](results/2026-10-27/design.json).

## Pendiente decisivo antes de usarlo como ERM en un paper

El contraejemplo `Z=I, V=I, alpha=(1,0), y=(0,2), C=1` da score multiplicativo cero y MSE 2.5, aunque `(1-epsilon,epsilon)` es factible y reduce el riesgo para epsilon positivo pequeño. No es siquiera un mínimo local de la frontera; la segunda coordenada queda bloqueada. Es un contraejemplo del solver abstracto y de la inferencia de estacionariedad a mínimo, no una refutación de todos los teoremas del artículo bajo todas sus hipótesis HAL.

No se han alterado silenciosamente las actualizaciones para activar ceros o sustituir la igualdad por una bola convexa. La siguiente decisión de método es resolver directamente la bola `||V alpha||1<=C` con un certificado de optimalidad, o ampliar y justificar el algoritmo de frontera para cruzar caras y activar PCs. Si la solución de la bola está activa, satisface también la igualdad; si es interior, ambos problemas son diferentes.

Otros pendientes separados: CV gaussiana SV construye la base con todas las X y reutiliza subfilas de U como inicializador; esas subfilas ya no son ortogonales. No se ha corregido ni validado como CV de un procedimiento entrenado íntegramente dentro de cada fold. Falta también decidir el control del intercepto y verificar aproximación sobre bolas SV fijas. No se ha modificado ningún manuscrito ni realizado commits, envíos o publicación.


## Later update: convex ball accepted

Carlos subsequently authorised the convex-ball solver. See [SV ball implementation and validation](../sv_ball/README.md). The historical results above are preserved. The historical stress/bridge scripts now call the explicit boundary entry points so that rerunning them does not silently use the new default. The previous local build's sources are archived with the new run; the old source hashes describe the state at that earlier execution.
