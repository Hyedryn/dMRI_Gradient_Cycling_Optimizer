import numpy as np
from . import optimizers # Use relative import
from . import cost       # Use relative import
from . import io         # Use relative import


def optimize_gradient_sequence(
    input_tensor_path,
    output_tensor_path,
    method='smart_brute_force',
    group_size=3,
    optimizer_kwargs=None
):
    """
    Optimizes the gradient sequence order in a GE tensor.dat file.

    Args:
        input_tensor_path (str): Path to the input tensor.dat file.
        output_tensor_path (str): Path to save the optimized tensor.dat file.
        method (str): Name of the optimization method to use
                      (e.g., 'smart_brute_force').
        group_size (int): TR group size (typically 2 or 3).
        optimizer_kwargs (dict, optional): Additional keyword arguments specific
                                           to the chosen optimizer
                                           (e.g., n_iter, N_to_permute). Defaults to None.

    Returns:
        np.ndarray: The optimized gradient vectors.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    print(f"Loading tensor file: {input_tensor_path}")
    vectors, n_dirs, header, all_lines = io.read_tensor_dat(input_tensor_path)
    print(f"Read {n_dirs} directions.")

    if n_dirs % group_size != 0:
         print(f"Warning: Number of directions ({n_dirs}) is not divisible by group size ({group_size}). Padding with b=0 vectors for optimization.")
         n_missing = group_size - (n_dirs % group_size)
         padding = np.zeros((n_missing, 3))
         padded_vectors = np.vstack((vectors, padding))
         print(f"Added {n_missing} padding vectors. Total vectors for optimization: {padded_vectors.shape[0]}")
    else:
         padded_vectors = vectors
         n_missing = 0

    base_cost, max_group_idx, _ = cost.eval_ge_cycling_cost(padded_vectors, group_size)
    print(f"Initial Max Cost: {base_cost:.4f} (Group Index: {max_group_idx})")

    print(f"Optimizing using method: {method} with group size: {group_size}")
    print(f"Optimizer arguments: {optimizer_kwargs}")

    # --- Select and run optimizer ---
    try:
        optimizer_func = getattr(optimizers, method)
    except AttributeError:
        raise ValueError(f"Unknown optimization method: {method}. Available methods: {dir(optimizers)}")

    # Pass group_size and other kwargs to the optimizer
    optimized_padded_vectors = optimizer_func(padded_vectors, group_size=group_size, **optimizer_kwargs)

    # Remove padding if it was added
    if n_missing > 0:
        optimized_vectors = optimized_padded_vectors[:-n_missing, :]
        print(f"Removed {n_missing} padding vectors.")
    else:
        optimized_vectors = optimized_padded_vectors


    # --- Verification ---
    if not np.allclose(np.sort(vectors.flatten()), np.sort(optimized_vectors.flatten())):
         print("WARNING: Vectors before and after optimization do not contain the same elements! Check optimizer logic.")
    else:
         print("Optimization complete. Vector elements verified.")

    final_cost, final_max_idx, _ = cost.eval_ge_cycling_cost(optimized_vectors, group_size)
    print(f"Final Max Cost: {final_cost:.4f} (Group Index: {final_max_idx})")


    print(f"Saving optimized tensor file to: {output_tensor_path}")
    io.write_tensor_dat(output_tensor_path, optimized_vectors, n_dirs, all_lines)

    return optimized_vectors