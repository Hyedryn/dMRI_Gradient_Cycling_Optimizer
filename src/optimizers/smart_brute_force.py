import numpy as np
import itertools
from tqdm import tqdm
from ..cost import eval_ge_cycling_cost
from math import factorial


def smart_brute_force(vectors, group_size=3, n_iter=10000, N_to_permute=6, weight_total_power=0, weight_adjacent_group=0):
    """
    Optimizes gradient order using a smarter brute-force approach.

    Focuses permutations on the group with the highest cost and a random
    selection of other vectors.

    Args:
        vectors (np.ndarray): Input vectors [N_directions, 3].
        group_size (int): TR group size (2 or 3).
        n_iter (int): Number of iterations.
        N_to_permute (int): Number of vectors to include in each permutation subset (4-8 recommended).
        weight_total_power (float): Weight for the total power term in cost function.
        weight_adjacent_group (float): Weight for the adjacent group term in cost function.

    Returns:
        np.ndarray: Optimized vector sequence.
    """

    current_permutation = np.copy(vectors)
    N_direction = current_permutation.shape[0]

    if N_direction == 0: return current_permutation # Handle empty input
    if N_direction % group_size != 0:
        raise ValueError(f"Number of directions ({N_direction}) must be divisible by group size ({group_size}).")

    N_groups = N_direction // group_size
    best_cost_function, max_group_idx, _ = eval_ge_cycling_cost(current_permutation, group_size, weight_total_power, weight_adjacent_group)
    print(f"Initial Max Cost: {best_cost_function:.4f} (Group Index: {max_group_idx})")

    # Indices that are not b=0 vectors
    indices_not_b0 = np.any(current_permutation != 0, axis=1)
    num_non_b0 = np.sum(indices_not_b0)

    if num_non_b0 <= N_to_permute:
        print("Warning: Number of non-b0 vectors is less than or equal to N_to_permute. Performing full permutation.")
        N_to_permute = num_non_b0
        if N_to_permute < 2:
             print("Warning: Fewer than 2 non-b0 vectors. No permutation possible.")
             return current_permutation

    print(f"Optimizing with N_to_permute = {N_to_permute}")
    n_perms_per_iter = factorial(N_to_permute)
    print(f"Permutations per iteration: {n_perms_per_iter}")


    pbar = tqdm(range(n_iter), desc="Smart Brute Force")
    for _ in pbar:
        # --- Identify indices in the worst group ---
        worst_group_indices = np.arange(max_group_idx * group_size, (max_group_idx + 1) * group_size)
        # Only consider non-b0 vectors within the worst group
        worst_group_non_b0_indices = worst_group_indices[indices_not_b0[worst_group_indices]]

        # --- Identify indices *not* in the worst group and *not* b0 ---
        other_non_b0_indices = np.where(indices_not_b0 & ~np.isin(np.arange(N_direction), worst_group_indices))[0]

        # --- Select indices for permutation ---
        num_from_worst = len(worst_group_non_b0_indices)
        num_from_others = N_to_permute - num_from_worst

        if num_from_others < 0: # N_to_permute is smaller than non-b0 in worst group
            # Select a subset from the worst group only
            indices_to_permute = np.random.choice(worst_group_non_b0_indices, N_to_permute, replace=False)
        elif len(other_non_b0_indices) < num_from_others:
            # Not enough 'other' vectors, take all 'others' and fill remaining from 'worst'
            num_to_take_from_worst = N_to_permute - len(other_non_b0_indices)
            indices_to_permute = np.concatenate([
                other_non_b0_indices,
                np.random.choice(worst_group_non_b0_indices, num_to_take_from_worst, replace=False)
            ])
        else:
            # Standard case: take all non-b0 from worst, fill rest from others
             indices_to_permute = np.concatenate([
                 worst_group_non_b0_indices,
                 np.random.choice(other_non_b0_indices, num_from_others, replace=False)
             ])

        indices_to_permute = np.sort(indices_to_permute) # Keep sorted for consistency if needed

        # --- Try all permutations of the selected subset ---
        best_perm_in_subset = np.copy(current_permutation) # Start with current best overall
        min_cost_in_subset = best_cost_function         # Current overall best cost

        original_subset_vectors = current_permutation[indices_to_permute]

        for perm in itertools.permutations(original_subset_vectors):
            temp_permutation = np.copy(current_permutation)
            temp_permutation[indices_to_permute] = perm # Apply permutation

            # Calculate cost for this specific permutation
            new_cost_function, new_max_group_idx, _ = eval_ge_cycling_cost(temp_permutation, group_size, weight_total_power, weight_adjacent_group)

            # Check if this permutation is better than the best found *in this subset*
            if new_cost_function < min_cost_in_subset:
                min_cost_in_subset = new_cost_function
                best_perm_in_subset = np.copy(temp_permutation) # Store the best permutation found
                max_group_idx = new_max_group_idx        # Update the worst group index globally

        # Update the overall best if the best from the subset search is better
        if min_cost_in_subset < best_cost_function:
             best_cost_function = min_cost_in_subset
             current_permutation = best_perm_in_subset # Update the main permutation for next iteration
             pbar.set_postfix({"Best Cost": f"{best_cost_function:.4f}"})


    print(f"\nFinal Max Cost: {best_cost_function:.4f} (Group Index: {max_group_idx})")
    return current_permutation
