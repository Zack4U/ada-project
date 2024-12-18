from datetime import datetime
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import random

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

def calculate_distributions(TPM, division):
    """Calculate the probability distributions for each subsystem."""
    group_1, group_2 = division
    if len(group_1) == 0 or len(group_2) == 0:
        raise ValueError("Los grupos no pueden estar vacíos")
    
    # Normalizar por columna para mantener propiedades de TPM
    dist_1 = TPM[group_1, :][:, group_1]
    dist_2 = TPM[group_2, :][:, group_2]
    
    # Suma por columnas y normalización
    dist_1_sum = dist_1.sum(axis=0)
    dist_2_sum = dist_2.sum(axis=0)
    
    return dist_1_sum / dist_1_sum.sum(), dist_2_sum / dist_2_sum.sum()

def calculate_emd(distribution1, distribution2):
    """Calculate the Earth Mover's Distance (EMD) between two distributions."""
    return wasserstein_distance(distribution1, distribution2)

def initialize_random_partition(num_states):
    """Initialize a random partition of states into two groups."""
    indices = list(range(num_states))
    random.shuffle(indices)
    split = num_states // 2
    return indices[:split], indices[split:]

def heuristic_partitioning(TPM, elements_t, max_iterations=1000, max_no_improve=100):
    """Heuristic partitioning of states to minimize EMD between distributions."""
    # Crear DataFrame para resultados
    results_df = pd.DataFrame(columns=['Timestamp', 'W0', 'W1', 'EMD', 'Tiempo_Verificacion'])
    
    num_states = len(elements_t)
    best_groups = initialize_random_partition(num_states)
    best_dist_1, best_dist_2 = calculate_distributions(TPM, best_groups)
    best_emd = wasserstein_distance(best_dist_1, best_dist_2)
    start1_time = time.time()
    
    print("\nIniciando búsqueda local iterativa:")
    print(f"Partición inicial:")
    print(f"W0: {best_groups[0]}")
    print(f"W1: {best_groups[1]}")
    print(f"EMD inicial: {best_emd}\n")
    
    no_improve = 0
    current_groups = best_groups
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        print(f"\nIteración {iteration + 1}/{max_iterations}")
        print("-" * 40)
        
        group_1, group_2 = current_groups
        if len(group_1) > 0 and len(group_2) > 0:
            idx1 = random.randint(0, len(group_1) - 1)
            idx2 = random.randint(0, len(group_2) - 1)
            
            new_group_1 = list(group_1)
            new_group_2 = list(group_2)
            new_group_1[idx1], new_group_2[idx2] = new_group_2[idx2], new_group_1[idx1]
            
            try:
                new_dist_1, new_dist_2 = calculate_distributions(TPM, (new_group_1, new_group_2))
                new_emd = wasserstein_distance(new_dist_1, new_dist_2)
                verification_time = time.time() - start_time
                
                # Registrar resultados
                results_df.loc[len(results_df)] = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'W0': str(new_group_1),
                    'W1': str(new_group_2),
                    'EMD': new_emd,
                    'Tiempo_Verificacion': verification_time
                }
                
                print(f"Evaluando partición:")
                print(f"W0: {new_group_1}")
                print(f"W1: {new_group_2}")
                print(f"EMD actual: {new_emd}")
                print(f"Mejor EMD hasta ahora: {best_emd}")
                
                if new_emd < best_emd:
                    print("¡Se encontró una mejor partición!")
                    best_groups = (new_group_1, new_group_2)
                    best_dist_1, best_dist_2 = new_dist_1, new_dist_2
                    best_emd = new_emd
                    no_improve = 0
                    
                    if best_emd == 0:
                        results_df.to_excel('resultados_busqueda_local.xlsx', index=False)
                        print("\n¡Se encontró partición óptima con EMD = 0!")
                        return best_groups[0], best_groups[1], best_emd
                else:
                    no_improve += 1
                    print(f"Sin mejora. Intentos sin mejorar: {no_improve}/{max_no_improve}")
                
                current_groups = (new_group_1, new_group_2)
                
                if no_improve >= max_no_improve:
                    print("\nSe alcanzó el límite de intentos sin mejora.")
                    break
                    
            except Exception as e:
                print(f"Error en iteración {iteration}: {e}")
                continue
    
    end_time1 = time.time()
    # Exportar resultados al finalizar
    results_df.to_excel('resultados_busqueda_local.xlsx', index=False)
    print("Resultados exportados a 'resultados_busqueda_local.xlsx'")
    return best_groups[0], best_groups[1], best_emd, start1_time, end_time1

def main():
    print("-" * 80)
    print("DIVISIÓN DE SISTEMAS - Búsqueda Local Iterativa")
    print("Analisis y Diseño de Algoritmos")
    print("-" * 80)
    
    # Leer el sistema desde JSON
    file_path = "red_11_nodos.json"
    result = read_system_from_json(file_path)
    
    if result is None:
        return
        
    elements_t, elements_t1, initial_state, TPM = result
    
    print(f"\nSistema cargado:")
    print(f"Elementos en t: {elements_t}")
    print(f"TPM shape: {TPM.shape}")
    print(f"Estado inicial: {initial_state}\n")
    
    W0, W1, emd, start_time, end_time = heuristic_partitioning(TPM, elements_t)
    
    print("\nResultados finales:")
    print(f"W0: {W0}")
    print(f"W1: {W1}")
    print(f"EMD: {emd}")
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

# Example usage
if __name__ == "__main__":
    main()