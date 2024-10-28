import numpy as np
import json

def get_system_input(prompt):
    """Obtiene una lista de variables ingresadas línea por línea"""
    print(prompt)
    variables = []
    while True:
        line = input().strip()
        if line == "":
            break
        variables.append(line)
    return variables

def get_initial_states(variables):
    """Obtiene los estados iniciales para cada variable"""
    print("\nIngrese valores iniciales")
    initial_states = {}
    for var in variables:
        while True:
            try:
                value = int(input(f"Para {var} = "))
                if value not in [0, 1]:
                    print("El valor debe ser 0 o 1")
                    continue
                initial_states[var] = value
                break
            except ValueError:
                print("Por favor ingrese un valor válido (0 o 1)")
    return initial_states

def generate_precise_tpm(n):
    """Genera una TPM precisa"""
    size = 2**n
    # Generar matriz aleatoria
    TPM = np.random.rand(size, size)
    
    # Normalizar cada fila para que sume 1
    row_sums = TPM.sum(axis=1)
    TPM = TPM / row_sums[:, np.newaxis]
    
    # Verificar propiedades básicas
    assert np.allclose(TPM.sum(axis=1), 1.0), "Las filas no suman 1"
    assert np.all(TPM >= 0) and np.all(TPM <= 1), "Valores fuera del rango [0,1]"
    
    print("\nPropiedades de la TPM generada:")
    print(f"- Dimensiones: {size}x{size}")
    print(f"- Rango de valores: [{TPM.min():.4f}, {TPM.max():.4f}]")
    
    return TPM.tolist()

def create_system_json():
    # Obtener sistema completo
    complete_system = get_system_input("\nIngrese Sistema completo")
    if not complete_system:
        print("Error: El sistema completo no puede estar vacío")
        return
    
    # Obtener sistema candidato
    candidate_system = get_system_input("\nIngrese sistema candidato")
    if not candidate_system:
        print("Error: El sistema candidato no puede estar vacío")
        return
    
    # Validar que el sistema candidato sea un subconjunto del sistema completo
    if not all(var in complete_system for var in candidate_system):
        print("Error: El sistema candidato debe ser un subconjunto del sistema completo")
        return
    
    # Obtener estados iniciales
    initial_states = get_initial_states(complete_system)
    
    # Generar elementos t+1
    elements = {
        "t": complete_system,
        "t+1": [f"{var}+1" for var in complete_system]
    }
    
    # Generar sistema candidato con t+1
    candidate = {
        "t": candidate_system,
        "t+1": [f"{var}+1" for var in candidate_system]
    }
    
    # Generar TPM
    TPM = generate_precise_tpm(len(complete_system))
    
    # Crear diccionario completo
    data = {
        "elements": elements,
        "candidate_system": candidate,
        "initial_state": [initial_states[var] for var in complete_system],
        "TPM": TPM
    }
    
    # Guardar resultado
    output_file = 'system_data.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\nDatos guardados exitosamente en '{output_file}'")
    print(f"Tamaño de la TPM generada: {len(TPM)}x{len(TPM[0])}")
    return data

if __name__ == "__main__":
    print("=== Generador de Sistema ===")
    create_system_json()