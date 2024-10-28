import numpy as np
import time
import itertools
from scipy.stats import wasserstein_distance
import json

"""DIVISION DE SISTEMAS
Hecho por: Juan Pablo Valencia Chaves y Kevin Santiago Lopez Salazar
Analisis y Diseño De Algoritmos
II Semestre 2024
"""
    
def emd(P_X, P_V):
    return wasserstein_distance(P_X, P_V)

def get_state_distribution(state_subset, n_elements, initial_state):
    """Calcula la distribución de estados para un subconjunto dado"""
    n_states = 2 ** n_elements
    distribution = np.zeros(n_states)
    
    if not state_subset:
        return np.ones(n_states) / n_states
    
    mask = np.zeros(n_elements, dtype=bool)
    for idx in state_subset:
        mask[idx] = True
    
    for state in range(n_states):
        binary_state = [int(x) for x in format(state, f'0{n_elements}b')]
        matches = True
        for i in range(n_elements):
            if mask[i] and binary_state[i] != initial_state[i]:
                matches = False
                break
        if matches:
            distribution[state] = 1
    
    return distribution / np.sum(distribution)

def marginalize_TPM(TPM, full_system, candidate_system, full_map):
    """Marginaliza la TPM para considerar solo el sistema candidato"""
    n_full = len(full_system['t'])
    n_candidate = len(candidate_system['t'])
    
    # Identificar índices de las variables a mantener
    keep_indices = [full_map['t'][elem] for elem in candidate_system['t']]
    
    # Crear nueva TPM marginalizada
    new_size = 2**n_candidate
    marginalized_TPM = np.zeros((new_size, new_size))
    
    # Para cada estado en el sistema completo
    for old_state in range(2**n_full):
        old_binary = [int(x) for x in format(old_state, f'0{n_full}b')]
        
        # Obtener el estado marginalizado
        new_binary = [old_binary[i] for i in keep_indices]
        new_state = int(''.join(map(str, new_binary)), 2)
        
        for old_next_state in range(2**n_full):
            old_next_binary = [int(x) for x in format(old_next_state, f'0{n_full}b')]
            new_next_binary = [old_next_binary[i] for i in keep_indices]
            new_next_state = int(''.join(map(str, new_next_binary)), 2)
            
            marginalized_TPM[new_state][new_next_state] += TPM[old_state][old_next_state]
    
    # Normalizar
    row_sums = marginalized_TPM.sum(axis=1)
    marginalized_TPM = marginalized_TPM / row_sums[:, np.newaxis]
    
    print("\nTPM original:")
    print(TPM)
    print("\nTPM marginalizada:")
    print(marginalized_TPM)
    
    
    return marginalized_TPM

def calculate_distribution(TPM, subset, state_map, initial_state):
    """Calcula la distribución de probabilidad sobre los estados"""
    n_elements = len(initial_state)
    state_subset = [state_map['t'][elem] for elem in subset if elem in state_map['t']]
    current_distribution = get_state_distribution(state_subset, n_elements, initial_state)
    next_distribution = np.dot(current_distribution, TPM)
    return next_distribution

def g(subset, TPM, V, state_map, initial_state):
    """Calcula el EMD entre las distribuciones"""
    P_X = calculate_distribution(TPM, subset, state_map, initial_state)
    P_V = calculate_distribution(TPM, V['t'], state_map, initial_state)
    return emd(P_X, P_V), P_X, P_V

def divide_system(TPM, full_system, candidate_system, initial_state):
    """Divide el sistema candidato en dos subsistemas"""
    full_map, candidate_map = create_state_mapping(full_system, candidate_system)
    
    # Marginalizar TPM y estado inicial
    marginalized_TPM = marginalize_TPM(TPM, full_system, candidate_system, full_map)
    candidate_initial_state = [initial_state[full_map['t'][elem]] 
                             for elem in candidate_system['t']]
    
    elements = candidate_system['t'].copy()
    n = len(elements)
    
    best_W0 = None
    best_W1 = None
    best_g = float('inf')
    best_P_X = None
    best_P_V = None
    
    # Probar todas las posibles divisiones
    for i in range(1, n):
        for W0 in itertools.combinations(elements, i):
            W0 = list(W0)
            W1 = [x for x in elements if x not in W0]
            
            current_g, P_X, P_V = g(W1, marginalized_TPM, candidate_system, 
                                  candidate_map, candidate_initial_state)
            
            if current_g < best_g:
                best_g = current_g
                best_W0 = W0
                best_W1 = W1
                best_P_X = P_X
                best_P_V = P_V
    
    return best_W0, best_W1, best_g, best_P_X, best_P_V

def verify_results(W0, W1, EMD, P_X, P_V, TPM, V):
    """Verifica los resultados"""
    verification = {
        "status": True,
        "messages": []
    }
    
    if not np.isclose(np.sum(P_X), 1.0, atol=1e-10):
        verification["status"] = False
        verification["messages"].append(f"P_X no suma 1: {np.sum(P_X)}")
    
    if not np.isclose(np.sum(P_V), 1.0, atol=1e-10):
        verification["status"] = False
        verification["messages"].append(f"P_V no suma 1: {np.sum(P_V)}")
    
    if not all(elem in V['t'] for elem in W0 + W1):
        verification["status"] = False
        verification["messages"].append("W0 o W1 contienen elementos inválidos")
    
    if set(W0).intersection(set(W1)):
        verification["status"] = False
        verification["messages"].append("W0 y W1 no son disjuntos")
    
    if set(W0 + W1) != set(V['t']):
        verification["status"] = False
        verification["messages"].append("W0 y W1 no cubren todos los elementos")
    
    return verification

def create_state_mapping(full_system, candidate_system):
    """Crea mapeos para el sistema completo y candidato"""
    full_map = {
        't': {elem: i for i, elem in enumerate(full_system['t'])},
        't+1': {elem: i for i, elem in enumerate(full_system['t+1'])}
    }
    
    candidate_map = {
        't': {elem: i for i, elem in enumerate(candidate_system['t'])},
        't+1': {elem: i for i, elem in enumerate(candidate_system['t+1'])}
    }
    
    return full_map, candidate_map

def main():
    
    print("-" *80)
    print("DIVISION DE SISTEMAS")
    print("Hecho por: Juan Pablo Valencia Chaves y Kevin Santiago Lopez Salazar")
    print("Analisis y Diseño De Algoritmos")
    print("II Semestre 2024")
    print("-" *80)
    
    
    
    try:
        # Leer datos de entrada
        #! (Cambiar nombre del archivo si es necesario)
        with open('system_data.json', 'r') as f:
            input_data = json.load(f)
        
        TPM = np.array(input_data['TPM'])
        full_system = input_data['elements']
        candidate_system = input_data['candidate_system']
        initial_state = np.array(input_data['initial_state'])
        
        print("\nSistema completo:", full_system)
        print("Sistema candidato:", candidate_system)
        print("Estado inicial:", initial_state)
        
        start_time = time.time()
        W0, W1, best_g, P_X, P_V = divide_system(TPM, full_system, 
                                                candidate_system, initial_state)
        end_time = time.time()
        
        print("\nResultados:")
        print(f"W0: {W0}")
        print(f"W1: {W1}")
        print(f"EMD: {best_g}")
        print(f"Tiempo de ejecución: {end_time - start_time} segundos")
        
        # Verificación
        verification = verify_results(W0, W1, best_g, P_X, P_V, TPM, candidate_system)
        if verification["status"]:
            print("\n✓ Todos los resultados son válidos")
        else:
            print("\n⚠ Se encontraron problemas:")
            for msg in verification["messages"]:
                print(f"  - {msg}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()