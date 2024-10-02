import numpy as np
from scipy.optimize import minimize
from numba import jit, prange
from multiprocessing import Pool
import time


@jit(nopython=True, parallel=True)
def calculate_reaction_quotient(concentrations, stoich_matrix_reactants, stoich_matrix_products, n_species,
                                n_reactions):
    Q_values = np.zeros(n_reactions)
    for j in prange(n_reactions):
        numerator = 1.0
        denominator = 1.0
        for i in range(n_species):
            if stoich_matrix_products[i, j] > 0:
                numerator *= concentrations[i] ** stoich_matrix_products[i, j]
            if stoich_matrix_reactants[i, j] > 0:
                denominator *= concentrations[i] ** stoich_matrix_reactants[i, j]
        Q_values[j] = numerator / denominator
    return Q_values


@jit(nopython=True)
def objective_function(reaction_progress, initial_concentrations, stoich_changes, stoich_matrix_reactants,
                       stoich_matrix_products, K_values, n_species, n_reactions):
    # Ensure all inputs are float64
    reaction_progress = reaction_progress.astype(np.float64)
    initial_concentrations = initial_concentrations.astype(np.float64)
    stoich_changes = stoich_changes.astype(np.float64)

    # Calculate concentrations
    concentrations = initial_concentrations + np.dot(stoich_changes, reaction_progress)

    # Check for negative concentrations
    if np.any(concentrations < 0):
        return 1e6 + np.sum(np.abs(concentrations[concentrations < 0]))

    # Calculate reaction quotients
    Q_values = calculate_reaction_quotient(concentrations, stoich_matrix_reactants, stoich_matrix_products, n_species,
                                           n_reactions)

    # Return the sum of squared errors between Q and K
    return np.sum((Q_values - K_values) ** 2)


def parallel_optimization(args):
    """Parallel wrapper for the optimization process."""
    (initial_guess, initial_concentrations, stoich_changes, stoich_matrix_reactants, stoich_matrix_products, K_values,
     n_species, n_reactions) = args

    # Use minimize to find the reaction progress that minimizes the objective function
    result = minimize(
        objective_function,
        initial_guess,
        args=(
        initial_concentrations, stoich_changes, stoich_matrix_reactants, stoich_matrix_products, K_values, n_species,
        n_reactions),
        method='SLSQP',
        bounds=[(None, None) for _ in range(len(initial_guess))],
        constraints=[{'type': 'ineq', 'fun': lambda x: initial_concentrations + np.dot(stoich_changes, x)}],
        tol=1e-12,
        options={'maxiter': 1000}
    )
    return result


def equilibrium_concentration(K_values, initial_concentrations, stoich_matrix_reactants, stoich_matrix_products):
    n_species = len(initial_concentrations)
    n_reactions = len(K_values)

    # Define the stoichiometric changes for each reaction and ensure type consistency
    stoich_changes = (stoich_matrix_products - stoich_matrix_reactants).astype(np.float64)

    # Initial guess for reaction progress (small values initially)
    initial_guess = np.zeros(n_reactions)

    # Parallel optimization using multiprocessing
    with Pool() as pool:
        args = [(initial_guess, initial_concentrations, stoich_changes, stoich_matrix_reactants, stoich_matrix_products,
                 K_values, n_species, n_reactions)]
        results = pool.map(parallel_optimization, args)

    # Extract the result from the parallel computation
    result = results[0]

    # Check if optimization was successful
    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    # Calculate equilibrium concentrations using the solved reaction progress
    reaction_progress_solution = result.x
    equilibrium_concentrations = initial_concentrations + np.dot(stoich_changes, reaction_progress_solution)

    return equilibrium_concentrations


# Example usage
if __name__ == "__main__":
    start_time = time.time()  # Start timing

    # Define equilibrium constants for the reactions
    K_values = np.array([4, 0.1])  # K1 = 4 for A + B + C -> D, K2 = 0.1 for D -> A + E

    # Define initial concentrations of 5 species (A, B, C, D, E)
    initial_concentrations = np.array([1.0, 1.0, 1.0, 2.0, 1.0])  # Initial concentrations in mol/L

    # Define stoichiometric matrices (5 species x 2 reactions)
    stoich_matrix_reactants = np.array([
        [1, 0],  # A in reactions 1 and 2
        [1, 0],  # B in reactions 1 and 2
        [1, 0],  # C in reactions 1 and 2
        [0, 1],  # D in reactions 1 and 2
        [0, 0],  # E in reactions 1 and 2
    ])
    stoich_matrix_products = np.array([
        [0, 1],  # A in reactions 1 and 2
        [0, 0],  # B in reactions 1 and 2
        [0, 0],  # C in reactions 1 and 2
        [1, 0],  # D in reactions 1 and 2
        [0, 1],  # E in reactions 1 and 2
    ])

    # Calculate equilibrium concentrations
    equilibrium_concs = equilibrium_concentration(K_values, initial_concentrations, stoich_matrix_reactants,
                                                  stoich_matrix_products)

    # End timing
    end_time = time.time()

    # Output results and timing
    print("Equilibrium concentrations:", equilibrium_concs)
    print(f"Time taken: {end_time - start_time:.4f} seconds")
