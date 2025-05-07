import numpy as np
import random
from tqdm import tqdm
from ..cost import eval_ge_cycling_cost # Use relative import

def _local_search_swap(current_vectors, group_size, weight_total_power, weight_adjacent_group, max_no_improve=50):
    """
    Performs a simple local search by randomly swapping non-b0 pairs.
    Stops after 'max_no_improve' swaps without finding a better solution.
    """
    best_vectors = np.copy(current_vectors)
    best_cost, _, _ = eval_ge_cycling_cost(best_vectors, group_size, weight_total_power, weight_adjacent_group)

    non_b0_indices = np.where(np.any(best_vectors != 0, axis=1))[0]
    if len(non_b0_indices) < 2:
        return best_vectors # Cannot swap if less than 2 non-b0 vectors

    no_improve_count = 0
    while no_improve_count < max_no_improve:
        # Select two distinct random non-b0 indices to swap
        idx1, idx2 = random.sample(list(non_b0_indices), 2)

        # Create temporary swapped version
        temp_vectors = np.copy(best_vectors)
        temp_vectors[[idx1, idx2]] = temp_vectors[[idx2, idx1]]

        # Evaluate cost
        current_cost, _, _ = eval_ge_cycling_cost(temp_vectors, group_size, weight_total_power, weight_adjacent_group)

        # If improvement found, accept the swap and reset counter
        if current_cost < best_cost:
            best_vectors = temp_vectors
            best_cost = current_cost
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_vectors


def _perturb_solution(vectors, perturbation_strength=5):
    """
    Applies a perturbation by performing a number of random swaps
    on non-b0 vectors.
    """
    perturbed_vectors = np.copy(vectors)
    non_b0_indices = np.where(np.any(perturbed_vectors != 0, axis=1))[0]

    if len(non_b0_indices) < 2:
        return perturbed_vectors # Cannot swap

    actual_strength = min(perturbation_strength, len(non_b0_indices) // 2) # Ensure enough pairs exist

    for _ in range(actual_strength):
        idx1, idx2 = random.sample(list(non_b0_indices), 2)
        perturbed_vectors[[idx1, idx2]] = perturbed_vectors[[idx2, idx1]]

    return perturbed_vectors


def iterated_local_search(
    vectors,
    group_size=3,
    n_iter=100, # Number of ILS iterations (Perturb + Local Search)
    local_search_depth=50, # Max swaps without improvement in local search
    perturbation_strength=5, # Number of random swaps in perturbation
    weight_total_power=0,
    weight_adjacent_group=0,
    ):
    """
    Optimizes gradient order using Iterated Local Search.

    Args:
        vectors (np.ndarray): Input vectors [N_directions, 3].
        group_size (int): TR group size (2 or 3).
        n_iter (int): Number of main ILS iterations.
        local_search_depth (int): Max swaps without improvement in local search phase.
        perturbation_strength (int): Number of random swaps during perturbation.
        weight_total_power (float): Weight for the total power term in cost function.
        weight_adjacent_group (float): Weight for the adjacent group term in cost function.

    Returns:
        np.ndarray: Optimized vector sequence.
    """
    print("Starting Iterated Local Search...")

    # 1. Initial Solution (can be the input or random)
    # Let's start with the input and improve it first
    print(" Performing initial local search...")
    current_best_vectors = _local_search_swap(vectors, group_size, weight_total_power, weight_adjacent_group, local_search_depth*2) # Deeper initial search
    current_best_cost, max_idx, _ = eval_ge_cycling_cost(current_best_vectors, group_size, weight_total_power, weight_adjacent_group)
    print(f" Initial cost after local search: {current_best_cost:.4f}")

    overall_best_vectors = np.copy(current_best_vectors)
    overall_best_cost = current_best_cost

    # 3. Main Loop
    pbar = tqdm(range(n_iter), desc="Iterated Local Search")
    for i in pbar:
        # a. Perturb the current best solution
        perturbed = _perturb_solution(current_best_vectors, perturbation_strength)

        # b. Local Search on perturbed solution
        locally_optimized = _local_search_swap(perturbed, group_size, weight_total_power, weight_adjacent_group, local_search_depth)
        new_cost, new_max_idx, _ = eval_ge_cycling_cost(locally_optimized, group_size, weight_total_power, weight_adjacent_group)

        # c. Acceptance Criterion (Accept if better than current best)
        if new_cost < current_best_cost:
            current_best_vectors = np.copy(locally_optimized)
            current_best_cost = new_cost
            # Update overall best if this is the best found so far
            if new_cost < overall_best_cost:
                overall_best_vectors = np.copy(locally_optimized)
                overall_best_cost = new_cost
                pbar.set_postfix({"Best Cost": f"{overall_best_cost:.4f}*", "Iter": i}) # Mark improvement
            else:
                 pbar.set_postfix({"Best Cost": f"{overall_best_cost:.4f}", "Iter": i})
        else:
             pbar.set_postfix({"Best Cost": f"{overall_best_cost:.4f}", "Iter": i})


    print(f"\nFinished ILS. Final Best Cost: {overall_best_cost:.4f}")
    return overall_best_vectors
