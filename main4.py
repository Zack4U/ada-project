from datetime import datetime
import itertools
import numpy as np
from scipy.stats import wasserstein_distance
import json
import time
import pandas as pd

def parse_input(data):
    """Parse the JSON input into structured data."""
    elements_t = data['elements']['t']
    elements_t1 = data['elements']['t+1']
    initial_state = np.array(data['initial_state'])
    TPM = np.array(data['TPM'])
    return elements_t, elements_t1, initial_state, TPM

def read_system_from_json(file_path):
    """Lee el sistema desde un archivo JSON"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return parse_input(data)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo {file_path} no es un JSON válido")
        return None
    except KeyError as e:
        print(f"Error: Falta la clave {e} en el JSON")
        return None

def calculate_distribution(TPM, subset, state_map, initial_state):
    """Calcula la distribución de probabilidad sobre los estados"""
    n_elements = len(initial_state)
    n_states = 2 ** n_elements
    
    # Obtener índices de estados del subconjunto
    state_subset = [state_map['t'][elem] for elem in subset if elem in state_map['t']]
    
    # Calcular distribución inicial
    current_distribution = np.zeros(len(TPM))  # Cambiar dimensión para coincidir con TPM
    
    # Mapear estados del subconjunto a la distribución
    for state in state_subset:
        if state < len(TPM):
            current_distribution[state] = 1.0
    
    # Normalizar la distribución
    if np.sum(current_distribution) > 0:
        current_distribution /= np.sum(current_distribution)
    
    # Calcular siguiente distribución
    next_distribution = np.dot(current_distribution, TPM)
    
    return next_distribution

def get_state_distribution(state_subset, n_elements, initial_state):
    """Calcula la distribución de estados para un subconjunto dado"""
    n_states = 2 ** n_elements
    distribution = np.zeros(n_states)
    
    if not state_subset:
        return distribution
    
    # Convertir a arrays de numpy del tipo correcto
    mask = np.zeros(n_elements, dtype=np.int32)
    initial_state = initial_state.astype(np.int32)
    
    for idx in state_subset:
        mask[idx] = 1
    
    for state in range(n_states):
        # Convertir el estado a representación binaria
        state_bits = np.array([int(b) for b in format(state, f'0{n_elements}b')], dtype=np.int32)
        
        # Comparar los bits relevantes según la máscara
        if np.all((state_bits & mask) == (initial_state & mask)):
            distribution[state] = 1.0
    
    # Normalizar la distribución
    if np.sum(distribution) > 0:
        distribution /= np.sum(distribution)
    
    return distribution

def brute_force_partitioning(TPM, elements_t, elements_t1, initial_state):
    """
    Implementa particionamiento por fuerza bruta evaluando todas las posibles combinaciones.
    """
    # Crear DataFrame para almacenar resultados
    results_df = pd.DataFrame(columns=['Timestamp', 'W0', 'W1', 'EMD', 'Tiempo_Verificacion'])
    
    n = len(elements_t)
    best_emd = float('inf')
    best_partition = None
    all_elements = set(elements_t)
    
    for i in range(1, n):
        for subset in itertools.combinations(elements_t, i):
            start_time = time.time()
            W0 = list(subset)
            W1 = list(all_elements - set(subset))
            
            try:
                state_map = {'t': {elem: idx for idx, elem in enumerate(elements_t)}}
                dist_W0 = calculate_distribution(TPM, W0, state_map, initial_state)
                dist_W1 = calculate_distribution(TPM, W1, state_map, initial_state)
                
                current_emd = wasserstein_distance(dist_W0, dist_W1)
                verification_time = time.time() - start_time
                
                # Guardar resultados en DataFrame
                new_row = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'W0': str(W0),
                    'W1': str(W1),
                    'EMD': current_emd,
                    'Tiempo_Verificacion': verification_time
                }
                results_df.loc[len(results_df)] = new_row
                
                print(f"Evaluando partición:")
                print(f"W0: {W0}")
                print(f"W1: {W1}")
                print(f"EMD actual: {current_emd}")
                
                if current_emd == 0:
                    # Exportar resultados antes de salir
                    results_df.to_excel('resultados_fuerza_bruta.xlsx', index=False)
                    print("Resultados exportados a 'resultados_particiones.xlsx'")
                    return W0, W1, current_emd
                
                if current_emd < best_emd:
                    best_emd = current_emd
                    best_partition = (W0, W1)
                    
            except Exception as e:
                print(f"Error al evaluar partición: {e}")
                continue
    
    # Exportar resultados al finalizar
    results_df.to_excel('resultados_particiones.xlsx', index=False)
    print("Resultados exportados a 'resultados_particiones.xlsx'")
    
    if best_partition is None:
        return [], [], float('inf')
    
    return best_partition[0], best_partition[1], best_emd

def main():
    print("-" * 80)
    print("DIVISION DE SISTEMAS - Fuerza Bruta")
    print("Hecho por: Juan Pablo Valencia Chaves y Kevin Santiago Lopez Salazar")
    print("Analisis y Diseño de Algoritmos")
    print("II Semestre 2024")
    print("-" * 80)
    
    # Leer sistema desde archivo JSON
    file_path = "red_11_nodos.json"
    result = read_system_from_json(file_path)
    
    if result is None:
        return
        
    elements_t, elements_t1, initial_state, TPM = result
    
    print(f"\nSistema cargado:")
    print(f"Elementos en t: {elements_t}")
    print(f"TPM shape: {TPM.shape}")
    print(f"Estado inicial: {initial_state}\n")
    
    start_time = time.time()
    W0, W1, emd = brute_force_partitioning(TPM, elements_t, elements_t1, initial_state)
    end_time = time.time()
    
    print("\nResultados finales:")
    print(f"Mejor W0: {W0}")
    print(f"Mejor W1: {W1}")
    print(f"Mejor EMD: {emd}")
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
    
    """# Ejemplo de uso
    elements = [0, 1, 2, 3]  # Ejemplo con 4 elementos
    TPM = np.array([[0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25]])  # Matriz de ejemplo
    initial_state = np.zeros(len(elements))
    
    start_time = time.time()
    W0, W1, emd = brute_force_partitioning(TPM, elements, elements, initial_state)
    end_time = time.time()
    
    print(f"Mejor partición encontrada:")
    print(f"W0: {W0}")
    print(f"W1: {W1}")
    print(f"EMD: {emd}")
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")"""

if __name__ == "__main__":
    main()