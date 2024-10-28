# División de Sistemas

Este proyecto implementa un algoritmo para dividir sistemas en subsistemas, utilizando la Distancia Earth Mover's (EMD) como métrica de evaluación.

## Navegación

- [Autores](#autores)
- [Descripción](#descripción)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del código](#estructura-del-codigo)
- [Salida](#salida)
- [Manejo de Errores](#manejo-de-errores)
- [Contribuciones](#contribuciones)
- [Referencias](#referencias)

## Autores

- Juan Pablo Valencia Chaves
- Kevin Santiago Lopez Salazar

## Descripción

El programa analiza y divide sistemas complejos en subsistemas más pequeños, utilizando matrices de transición de probabilidad (TPM) y distribuciones de estado. El algoritmo encuentra la división óptima que minimiza la distancia EMD entre las distribuciones de probabilidad resultantes.

## Características

- Marginalización de matrices TPM
- Cálculo de distribuciones de estado  
- Evaluación de todas las posibles divisiones de sistemas
- Verificación de resultados
- Manejo de mapeos de estado

## Requisitos

- Python 3.x
- NumPy
- SciPy
- itertools (incluido en Python estándar)
- json (incluido en Python estándar)

## Instalación

```bash
pip install numpy scipy
```

## Uso

1. Prepare un archivo system_data.json con la siguiente estructura:

```bash
{
    "TPM": [[...]], 
    "elements": {
        "t": [...],
        "t+1": [...]
    },
    "candidate_system": {
        "t": [...],
        "t+1": [...]
    },
    "initial_state": [...]
}
```

... O genere uno automaticamente ejecutando:

```bash
python generate_data.py
```

2. Ejecute el programa:

```bash
python main.py
```

## Estructura del codigo

### Funciones Principales

`emd(P_X, P_V)`
Calcula la distancia Earth Movers entre dos distribuciones de probabilidad.
Parámetros:
P_X: Primera distribución
P_V: Segunda distribución
Retorna: Valor EMD

`get_state_distribution(state_subset, n_elements, initial_state)`
Calcula la distribución de estados para un subconjunto dado.
Parámetros:
state_subset: Subconjunto de estados
n_elements: Número total de elementos
initial_state: Estado inicial del sistema
Retorna: Distribución de estados normalizada

`marginalize_TPM(TPM, full_system, candidate_system, full_map)`
Marginaliza la matriz de transición para considerar solo el sistema candidato.
Parámetros:
TPM: Matriz de transición original
full_system: Sistema completo
candidate_system: Sistema candidato
full_map: Mapeo del sistema completo
Retorna: TPM marginalizada

`calculate_distribution(TPM, subset, state_map, initial_state)`
Calcula la distribución de probabilidad sobre los estados.
Parámetros:
TPM: Matriz de transición
subset: Subconjunto de estados
state_map: Mapeo de estados
initial_state: Estado inicial
Retorna: Distribución de probabilidad

`divide_system(TPM, full_system, candidate_system, initial_state)`
Función principal que divide el sistema candidato en dos subsistemas.
Parámetros:
TPM: Matriz de transición
full_system: Sistema completo
candidate_system: Sistema candidato
initial_state: Estado inicial
Retorna: W0, W1, mejor_g, P_X, P_V

`verify_results(W0, W1, EMD, P_X, P_V, TPM, V)`
Verifica la validez de los resultados obtenidos.
Parámetros:
W0, W1: Subsistemas resultantes
EMD: Valor EMD calculado
P_X, P_V: Distribuciones de probabilidad
TPM: Matriz de transición
V: Sistema candidato
Retorna: Diccionario con estado de verificación y mensajes

## Salida

```bash
----------------------------------------
DIVISION DE SISTEMAS
Hecho por: Juan Pablo Valencia Chaves y Kevin Santiago Lopez Salazar
Analisis y Diseño De Algoritmos
II Semestre 2024
----------------------------------------
Sistema completo: {...}
Sistema candidato: {...}
Estado inicial: [...]
TPM original:
[...]
TPM marginalizada:
[...]
Resultados:
W0: [...]
W1: [...]
EMD: ...
Tiempo de ejecución: ... segundos
✓ Todos los resultados son válidos
```

## Manejo de Errores

El programa incluye manejo de excepciones para:

1. Errores en la lectura del archivo JSON
2. Problemas de formato en los datos de entrada
3. Errores en cálculos matemáticos

## Contribuciones

Este proyecto fue desarrollado como parte del curso de Análisis y Diseño de Algoritmos, II Semestre 2024.

## Referencias

1. Documentación de NumPy: https://numpy.org/doc/
2. Documentación de SciPy: https://docs.scipy.org/doc/
3. Earth Mover's Distance: Wikipedia
