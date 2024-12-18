import json
import numpy as np
from scipy.stats import wasserstein_distance
import time
import pandas as pd
from datetime import datetime

"""DIVISION DE SISTEMAS #2 Estrategia
Hecho por: Juan Pablo Valencia Chaves y Kevin Santiago Lopez Salazar
Analisis y Diseño De Algoritmos
II Semestre 2024
"""

def marginalize_tpm(tpm, edges, edge_to_remove):
    """
    Marginaliza las probabilidades eliminando las conexiones especificadas.
    tpm: Matriz de probabilidades.
    edges: Lista de aristas (strings).
    edge_to_remove: Arista que se debe eliminar.
    Devuelve una TPM ajustada.
    """
    # Obtener el índice de la arista a eliminar
    try:
        index_to_remove = edges.index(edge_to_remove)
    except ValueError:
        raise ValueError(f"La arista {edge_to_remove} no se encuentra en la lista de aristas.")
    
    # Eliminar la fila y columna correspondiente
    marginal_tpm = np.delete(tpm, index_to_remove, axis=0)  # Filas
    marginal_tpm = np.delete(marginal_tpm, index_to_remove, axis=1)  # Columnas
    
    # También eliminar la arista de la lista de aristas
    edges.pop(index_to_remove)
    
    return marginal_tpm, edges


def calculate_emd(tpm_full, tpm_partial):
    """
    Calcula la distancia EMD entre la distribución completa y la parcial.
    """
    full_flat = tpm_full.flatten()
    partial_flat = tpm_partial.flatten()
    return wasserstein_distance(full_flat, partial_flat)


def run_algorithm(data):
    
    # Parse JSON input
    elements_t = data["elements"]["t"]
    elements_t1 = data["elements"]["t+1"]
    candidate_t = data["candidate_system"]["t"]
    candidate_t1 = data["candidate_system"]["t+1"]
    initial_state = data["initial_state"]
    tpm = np.array(data["TPM"])
    
    print("=== Datos cargados ===")
    print(f"Elementos en t (minusculas): {elements_t}")
    print(f"Elementos en t+1 (mayusculas): {elements_t1}")
    print(f"Sistema candidato en t: {candidate_t}")
    print(f"Sistema candidato en t+1: {candidate_t1}")
    print(f"Estado inicial del sistema: {initial_state}")
    
    # Conjunto de aristas iniciales
    edges = [f"{i}{j}" for i in candidate_t for j in candidate_t1]
    print(f"Conjunto inicial de aristas: {edges}")
    
    # Crear DataFrame para almacenar resultados
    results_df = pd.DataFrame(columns=['Timestamp', 'Edge', 'EMD', 'Tiempo_Verificacion'])
    
    # Inicia el proceso del macroalgoritmo
    A = edges.copy()  # Hacemos una copia para evitar modificar la lista original
    partitions = []
    min_loss_partition = None
    min_loss = float('inf')
    
    iteration = 0
    while len(A) > 1:
        start_time1 = time.time()
        iteration += 1
        print(f"\n=== Iteracion {iteration} ===")
        print(f"Aristas actuales: {A}")
        
        # Proceso de selección de la mejor partición (argmin del EMD)
        best_loss = float('inf')
        best_edge = None
        for edge in A:
            start_time = time.time()
            
            # Generar TPM sin la conexión actual
            marginal_tpm, _ = marginalize_tpm(tpm.copy(), A.copy(), edge)
            emd_loss = calculate_emd(tpm, marginal_tpm)
            verification_time = time.time() - start_time
            
            # Guardar resultados en DataFrame
            new_row = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Edge': edge,
                'EMD': emd_loss,
                'Tiempo_Verificacion': verification_time
            }
            results_df.loc[len(results_df)] = new_row
        
            print(f"Evaluando particion eliminando {edge}: perdida EMD = {emd_loss}")
            
            if emd_loss < best_loss:
                best_loss = emd_loss
                best_edge = edge
        
        print(f"Mejor particion en esta iteracion: {best_edge} con perdida = {best_loss}")
        
        # Verificar si genera partición (en este caso guardamos información sobre la partición)
        partitions.append((best_edge, best_loss))
        if best_loss < min_loss:
            min_loss = best_loss
            min_loss_partition = best_edge
        
        # Eliminar la mejor conexión del conjunto
        tpm, A = marginalize_tpm(tpm, A, best_edge)
        print(f"Eliminando {best_edge}, nuevas aristas: {A}")
        
        break
    
    end_time = time.time()
    print("\n=== Evaluacion final ===")
    print(f"Mejor partición: {min_loss_partition}")
    print(f"Mejor EMD: {min_loss}")
    
    end_time = time.time()
    print(f"Tiempo total de ejecucion: {end_time - start_time1:.4f} segundos")
    results_df.to_excel('resultados_voraz.xlsx', index=False)
    print("Resultados exportados a 'resultados_voraz.xlsx'")


if __name__ == "__main__":
    print("-" *80)
    print("DIVISION DE SISTEMAS - Estrategia #2")
    print("Hecho por: Juan Pablo Valencia Chaves y Kevin Santiago Lopez Salazar")
    print("Analisis y Diseño De Algoritmos")
    print("II Semestre 2024")
    print("-" *80)
    try:
        with open("red_11_nodos.json", "r") as file:
            data = json.load(file)
            run_algorithm(data)
    except Exception as e:
        print(f"Error: {e}")
