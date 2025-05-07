import numpy as np

def eval_ge_cycling_cost(tensor_in, group_size, weight_total_power=0, weight_adjacent_group=0):
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
        weight_adjacent_group (float): Optional weight for penalizing interactions
                                     between adjacent groups. Defaults to 0.

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
    all_cost_function = np.zeros(N_groups) # Shape (N_groups,)

    # Compute sum of absolute dot products between pairs within each group
    for m in range(group_size - 1): # Iterate through pairs in the group
        for n in range(m + 1, group_size):
            # Sum dot products across x, y, z for each group
            dot_products = np.sum(group_tensor[:, m, :] * group_tensor[:, n, :], axis=0)
            all_cost_function += np.abs(dot_products)

    # Add weighted total power term if specified
    if weight_total_power > 0:
        # Sum of squares across x,y,z -> sum across group members -> sum across groups
        total_power_per_group = np.sum(np.sum(group_tensor ** 2, axis=0), axis=0)
        all_cost_function += weight_total_power * total_power_per_group
    
        
    # Add weighted adjacent group interaction term if specified (VECTORIZED)
    if weight_adjacent_group > 0 and N_groups > 1:
        # T1 contains groups 0 to N_groups-2. Shape: (3, group_size, N_groups-1)
        T1 = group_tensor[:, :, :-1]
        # T2 contains groups 1 to N_groups-1. Shape: (3, group_size, N_groups-1)
        T2 = group_tensor[:, :, 1:]

        # Calculate pairwise dot products between vectors of adjacent groups using einsum
        # 'xmi,xni->mni': sum over 'x' (xyz), keep 'm' (vec idx in group i),
        # 'n' (vec idx in group i+1), and 'i' (adjacent pair index)
        # Result shape: (group_size, group_size, N_groups-1)
        dot_prods_adjacent_pairs = np.einsum('xmi,xni->mni', T1, T2, optimize='optimal')

        # Sum of absolute dot products for each adjacent pair (i, i+1)
        # Sum over axes 0 ('m') and 1 ('n'). Result shape: (N_groups-1,)
        interaction_sum_per_pair = np.sum(np.abs(dot_prods_adjacent_pairs), axis=(0, 1))

        # Initialize cost contribution from adjacent interactions
        adjacent_group_cost_contribution = np.zeros(N_groups)

        # Add interaction sum S[i] = sum(|dot(vec_group_i, vec_group_i+1)|)
        # to the cost of group i (affecting cost[0]...cost[N-2])
        adjacent_group_cost_contribution[:-1] += interaction_sum_per_pair
        # Add interaction sum S[i] also to the cost of group i+1 (affecting cost[1]...cost[N-1])
        adjacent_group_cost_contribution[1:]  += interaction_sum_per_pair
        
        # Add the weighted cost to the total cost function
        all_cost_function += weight_adjacent_group * adjacent_group_cost_contribution
        
    # Normalize cost
    all_cost_function = all_cost_function/(1 + weight_total_power + weight_adjacent_group)
        
    if all_cost_function.size == 0: # Handle case with zero groups
         max_cost = 0.0
         max_index = 0
    else:
        max_cost = np.max(all_cost_function)
        max_index = np.argmax(all_cost_function)

    return max_cost, max_index, all_cost_function