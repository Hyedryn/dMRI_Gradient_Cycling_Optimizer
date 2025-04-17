import numpy as np

def eval_ge_cycling_cost(tensor_in, group_size, weight_total_power=0):
    """
    Evaluates the group cycling cost function for 2TR or 3TR optimization.

    This cost function reflects the interaction between gradients applied
    within the same TR block, aiming to minimize effects that contribute
    to thermal load.

    Args:
        tensor_in (np.ndarray): Input tensor of shape [N_directions, 3].
        group_size (int): The number of TRs in a cycle (e.g., 2 or 3).
        weight_total_power (float): Optional weight for the total power term.
                                     Typically 0, but can be used for multi-shell.

    Returns:
        tuple: A tuple containing:
            - float: The maximum cost function value across all groups.
            - int: The index of the group with the maximum cost.
            - np.ndarray: Array of cost function values for all groups.
    """

    N_direction = tensor_in.shape[0]
    if N_direction == 0:
        return 0.0, 0, np.array([0.0])
    if N_direction % group_size != 0:
        raise ValueError(f"The number of directions ({N_direction}) must be divisible by the group size ({group_size})!")

    N_groups = N_direction // group_size
    # Reshape to [3, group_size, N_groups]
    group_tensor = tensor_in.T.reshape((3, group_size, N_groups), order="F")

    dim = group_tensor.shape

    # Initialize cost function array for all groups
    all_cost_function = np.zeros(dim[2]) # Shape (N_groups,)

    # Compute sum of absolute dot products between pairs within each group
    for m in range(dim[1] - 1): # Iterate through pairs in the group
        for n in range(m + 1, dim[1]):
            # Sum dot products across x, y, z for each group
            dot_products = np.sum(group_tensor[:, m, :] * group_tensor[:, n, :], axis=0)
            all_cost_function += np.abs(dot_products)

    # Add weighted total power term if specified
    if weight_total_power > 0:
        # Sum of squares across x,y,z -> sum across group members -> sum across groups
        total_power_per_group = np.sum(np.sum(group_tensor ** 2, axis=0), axis=0)
        all_cost_function += weight_total_power * total_power_per_group

    if all_cost_function.size == 0: # Handle case with zero groups
         max_cost = 0.0
         max_index = 0
    else:
        max_cost = np.max(all_cost_function)
        max_index = np.argmax(all_cost_function)

    return max_cost, max_index, all_cost_function