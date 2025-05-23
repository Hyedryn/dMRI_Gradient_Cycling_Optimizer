import numpy as np
import math
import datetime
import os


def read_tensor_dat(file_path, n_dirs_selected=None):
    """
    Reads a GE tensor.dat file.
    If n_dirs_selected is None, it extracts the scheme with the largest number of directions.
    If n_dirs_selected is an integer, it attempts to extract the first scheme found
    with exactly that many directions.

    Args:
        file_path (str): Path to the tensor.dat file.
        n_dirs_selected (int, optional): If specified, look for a block with
                                         this specific number of directions.
                                         Defaults to None (find largest block).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of gradient vectors [N_directions, 3] for the selected/largest scheme.
            - int: Number of directions in the selected/largest scheme found.
            - list: Header lines from the file.
            - list: All lines read from the file (for reconstruction).
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Failed to open the tensor file: {file_path}")

    all_lines = [line for line in lines]  # Keep original lines with endings
    header_lines = [line for line in lines if line.strip().startswith("#")]

    blocks_found = []  # Store tuples: (num_dirs, start_line_index_of_num_dirs_line, vector_start_line_index)
    current_line_idx = 0

    # --- Pass 1: Identify all potential blocks, their sizes, and their start indices ---
    while current_line_idx < len(lines):
        line_content_stripped = lines[current_line_idx].strip()

        # Skip headers and empty lines
        if line_content_stripped.startswith("#") or not line_content_stripped:
            current_line_idx += 1
            continue

        # Check if this line indicates the number of directions for a block
        try:
            num_directions_in_block = int(line_content_stripped)
            if num_directions_in_block <= 0:
                # This could be a negative number or zero, which isn't a valid count.
                # It might be part of a vector if parsing went wrong, or just invalid.
                # For robustness, let's assume it's not a block start and move on.
                # A more strict parser might raise an error here.
                current_line_idx += 1
                continue

            # Store the number of directions, the index of THIS line,
            # and the index of the *next* line (start of vectors)
            vector_data_start_index = current_line_idx + 1
            blocks_found.append((num_directions_in_block, current_line_idx, vector_data_start_index))

            # Skip past the vector lines for this block to find the next block size indicator
            current_line_idx += num_directions_in_block + 1

        except ValueError:
            # This line is not a valid number of directions indicator.
            # It could be a vector line from a previous block if the count was off,
            # or just a malformed line.
            current_line_idx += 1
        except IndexError:
            # Reached end of file prematurely after reading num_directions
            break

    if not blocks_found:
        raise ValueError("No valid direction blocks found in the tensor file.")

    # --- Select the target block ---
    target_block_info = None
    if n_dirs_selected is not None:
        # User specified a number of directions to find
        for block in blocks_found:
            if block[0] == n_dirs_selected:
                target_block_info = block
                break # Found the first matching block
        if target_block_info is None:
            raise ValueError(f"No block found with exactly {n_dirs_selected} directions. "
                             f"Available block sizes: {[b[0] for b in blocks_found]}")
    else:
        # User did not specify, find the largest block
        if not blocks_found: # Should be caught above, but defensive
             raise ValueError("No blocks found to determine the largest.")
        target_block_info = max(blocks_found, key=lambda item: item[0])

    actual_num_directions, _, vector_start_idx_for_extraction = target_block_info

    # --- Pass 2: Extract vectors for the selected block ---
    tensors = []
    vectors_read_count = 0

    if vector_start_idx_for_extraction >= len(lines):
        raise ValueError(
            f"Tensor file format error: Declared vector start index "
            f"{vector_start_idx_for_extraction} for block with {actual_num_directions} "
            f"directions is out of bounds (file has {len(lines)} lines).")

    # Read exactly 'actual_num_directions' vectors starting from 'vector_start_idx_for_extraction'
    for i in range(vector_start_idx_for_extraction, len(lines)):
        if vectors_read_count >= actual_num_directions:
            break  # Stop after reading the expected number for THIS block

        line_content_stripped = lines[i].strip()
        if not line_content_stripped and vectors_read_count < actual_num_directions:
            # An empty line within an expected vector block could be problematic.
            # Depending on strictness, one might raise an error or issue a warning.
            # For now, let's skip it and see if we still get enough vectors.
            # This could lead to a warning/error later if not enough vectors are read.
            print(f"Warning: Encountered empty line at line index {i} within expected vector data for block "
                  f"of size {actual_num_directions}. Skipping.")
            continue

        parts = line_content_stripped.split()
        if len(parts) == 3:
            try:
                tensors.append([float(p) for p in parts])
                vectors_read_count += 1
            except ValueError:
                raise ValueError(
                    f"Invalid vector data (non-numeric) found at line {i + 1} (index {i}) "
                    f"in block expecting {actual_num_directions} directions: '{line_content_stripped}'")
        else:
            # Found a line that isn't 3 components long where a vector was expected.
            # This indicates either the end of the block or a malformed file.
            print(
                f"Warning: Expected 3 components for a vector at line {i + 1} (index {i}) "
                f"but found {len(parts)} components: '{line_content_stripped}'. "
                f"Stopping read for this block of {actual_num_directions} directions "
                f"after reading {vectors_read_count} vectors.")
            break # Assume end of this specific vector block

    if vectors_read_count != actual_num_directions:
        print(
            f"Warning: Expected {actual_num_directions} directions for the selected block, "
            f"but only read {vectors_read_count}. This might occur if the block is "
            f"truncated or contains malformed lines. "
            f"File: '{file_path}', expected block started at line index {vector_start_idx_for_extraction-1}.")
        # Adjust actual_num_directions to what was actually read for this block
        # This is important so that the returned number of directions matches the tensor_np shape.
        actual_num_directions = vectors_read_count

    tensors_np = np.array(tensors)

    # Final check on the shape of the extracted tensor data
    if tensors_np.ndim == 1 and actual_num_directions == 0 : # Handle case of empty block successfully read as empty
         tensors_np = tensors_np.reshape(0,3) # ensure it's 2D [0,3]
    elif tensors_np.shape != (actual_num_directions, 3):
        raise ValueError(
            f"Shape mismatch for the extracted block: Expected ({actual_num_directions}, 3) "
            f"based on (potentially adjusted) count, but got {tensors_np.shape}. "
            f"This indicates a severe parsing issue or file corruption for the selected block.")

    return tensors_np, actual_num_directions, header_lines, all_lines

def write_tensor_dat(output_path, optimized_tensors, original_num_directions, all_original_lines):
    """
    Writes the optimized tensors back into the tensor.dat format,
    preserving the original structure.

    Args:
        output_path (str): Path to save the new tensor.dat file.
        optimized_tensors (np.ndarray): The reordered gradient vectors.
        original_num_directions (int): The number of directions from the original file.
        all_original_lines (list): All lines from the original file.
    """
    if optimized_tensors.shape[0] != original_num_directions:
        raise ValueError("Number of optimized tensors does not match original number of directions.")
    if optimized_tensors.shape[1] != 3:
        raise ValueError("Optimized tensors must have 3 columns (x, y, z).")

    output_lines = []
    vector_idx = 0
    in_vector_block = False

    # Find the start of the vector block corresponding to original_num_directions
    line_idx_num_dirs = -1
    for i, line in enumerate(all_original_lines):
         if not line.strip().startswith("#") and line.strip():
             try:
                 num_dirs_in_line = int(line.strip())
                 if num_dirs_in_line == original_num_directions:
                      line_idx_num_dirs = i
                      break
             except ValueError:
                 continue

    if line_idx_num_dirs == -1:
        raise ValueError(f"Could not find the block for {original_num_directions} directions in original file lines.")


    vector_start_idx = line_idx_num_dirs + 1

    for i, line in enumerate(all_original_lines):
        if i == line_idx_num_dirs:
            output_lines.append(line) # Add the line with the number of directions
            in_vector_block = True
        elif in_vector_block and vector_idx < original_num_directions:
             # Replace this line with the optimized vector
            vec = optimized_tensors[vector_idx]
            # Use formatting that matches typical tensor.dat files
            output_lines.append(f"  {vec[0]: 18.15f}  {vec[1]: 18.15f}  {vec[2]: 18.15f}\n")
            vector_idx += 1
            # Check if the *next* line indicates the end of the block
            if vector_idx == original_num_directions:
                in_vector_block = False
        elif not in_vector_block:
             # Copy other lines (headers, other potential blocks)
             output_lines.append(line)
        # Else (inside vector block but already written all vectors), skip original vector lines


    try:
        with open(output_path, 'w') as file:
            file.writelines(output_lines)
    except IOError as e:
        raise IOError(f"Failed to write optimized tensor file to {output_path}: {e}")

def format_ge_tensor_for_sequence(bvalues, vectors_per_shell, b0_count, output_prefix, b0_spacing=0):
    """
    Generates the tensor.dat file in GE format from generated samples.

    Args:
        bvalues (list): List of b-values for each shell.
        vectors_per_shell (list): List of number of vectors for each shell.
        b0_count (int): Total number of b=0 volumes.
        output_prefix (str): Base path and filename prefix for output files.
        b0_spacing (int): Interval at which to insert b=0 volumes (0 for none).

    Returns:
        str: The path to the generated tensor.dat file.
    """
    samples_file_path = f"{output_prefix}_samples.txt"
    if not os.path.exists(samples_file_path):
         # Attempt to generate samples if file doesn't exist (requires qspace)
         print(f"Samples file {samples_file_path} not found. Attempting generation...")
         try:
             from qspace.sampling import multishell as ms # Lazy import
             from qspace.visu import visu_points
             from matplotlib import pyplot as plt
             gen_samples(vectors_per_shell, output_prefix) # Call the generation part
             print("Sample generation complete.")
         except ImportError:
              raise ImportError("The 'qspace' library is required to generate samples. Please install it or provide a pre-generated _samples.txt file.")
         except Exception as e:
              raise RuntimeError(f"Failed to generate samples: {e}")


    with open(samples_file_path, 'r') as sf:
        lines = sf.readlines()

    shells, u_x, u_y, u_z = [], [], [], []
    for l in lines:
        if l.strip().startswith('#') or not l.strip():
            continue
        data = l.split()
        shells.append(int(data[0]))
        u_x.append(float(data[1]))
        u_y.append(float(data[2]))
        u_z.append(float(data[3]))

    n_shells = len(set(shells))
    n_dir_total = len(shells)
    if len(bvalues) != n_shells:
        raise ValueError(f"Mismatch: {n_shells} shells in samples file, {len(bvalues)} b-values provided.")

    # Scale gradients
    b_max = float(max(bvalues)) if bvalues else 1.0 # Avoid division by zero if no b>0 shells
    scaled_u_x, scaled_u_y, scaled_u_z = [], [], []
    original_indices = list(range(n_dir_total)) # Keep track of original order

    for i in range(n_dir_total):
        shell_idx = shells[i]
        if shell_idx < 0 or shell_idx >= len(bvalues):
            raise IndexError(f"Invalid shell index {shells[i]} encountered.")
        b = bvalues[shell_idx] / b_max if b_max > 0 else 0.0
        norm = math.sqrt(u_x[i]**2 + u_y[i]**2 + u_z[i]**2)
        if norm == 0: norm = 1.0 # Avoid division by zero for potential b0 in samples

        # Apply scaling and GE's x-gradient inversion
        sqrt_b_over_norm = math.sqrt(b) / norm
        scaled_u_x.append(-u_x[i] * sqrt_b_over_norm)
        scaled_u_y.append( u_y[i] * sqrt_b_over_norm)
        scaled_u_z.append( u_z[i] * sqrt_b_over_norm)

    # Prepare header
    tensor_dat_path = f"{output_prefix}_tensor.dat"
    header = [
        "# Multi-shell tensor file generated by dMRI_Gradient_Cycling_Optimizer",
        f"# Date generated: {datetime.datetime.now()}",
        "#",
        "# Multi-shell details:",
        f"# - DWI shell 0: b value = 0; directions = {b0_count}"
    ]
    total_dwi_dirs = 0
    shell_counts_actual = {}
    for i in range(n_dir_total):
         shell_counts_actual[shells[i]] = shell_counts_actual.get(shells[i], 0) + 1

    for i in range(n_shells):
        num_dirs_in_shell = shell_counts_actual.get(i, 0) # Use actual count from file
        header.append(f"# - DWI shell {i + 1}: b value = {bvalues[i]}; directions = {num_dirs_in_shell}")
        total_dwi_dirs += num_dirs_in_shell
    header.append("#")

    # Write the tensor.dat file with interleaving
    with open(tensor_dat_path, 'w') as f:
        f.write("\n".join(header) + "\n")

        # Write initial 6 b0 lines for GE compatibility
        f.write('6\n')
        for _ in range(6):
            f.write('  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00\n')

        # Write the main block size
        total_lines_in_block = total_dwi_dirs + b0_count
        f.write(f'{total_lines_in_block}\n')

        b0_inserted = 0
        current_dwi_idx = 0
        for i in range(total_lines_in_block):
            # Insert b0 at specified spacing, ensuring we don't exceed b0_count
            should_insert_b0 = (b0_spacing != 0 and i % (b0_spacing + 1) == 0 and b0_inserted < b0_count)
            # Or insert remaining b0s at the end if spacing didn't cover all
            is_end_padding_b0 = (current_dwi_idx >= total_dwi_dirs and b0_inserted < b0_count)


            if should_insert_b0 or is_end_padding_b0:
                f.write('  0.000000000000000e+00  0.000000000000000e+00  0.000000000000000e+00\n')
                b0_inserted += 1
            elif current_dwi_idx < total_dwi_dirs:
                 # Write the next DWI vector
                 orig_idx = original_indices[current_dwi_idx]
                 f.write(f"  {scaled_u_x[orig_idx]: 18.15f}  {scaled_u_y[orig_idx]: 18.15f}  {scaled_u_z[orig_idx]: 18.15f}\n")
                 current_dwi_idx += 1
            # Else: we've written all DWIs and all B0s, should not happen if total_lines_in_block is correct

        f.write('\n') # Add a newline at the end

    print(f"Generated GE tensor file: {tensor_dat_path}")
    print(f"Total directions (DWI + b0): {total_dwi_dirs + b0_count}")
    return tensor_dat_path


def gen_samples(points_per_shell, output):
    """Generates diffusion sampling points using qspace library."""
    try:
        from qspace.sampling import multishell as ms # Lazy import
        from qspace.visu import visu_points
        from matplotlib import pyplot as plt
    except ImportError:
        raise ImportError("The 'qspace' and 'matplotlib' libraries are required to generate samples.")

    nb_shells = len(points_per_shell)
    K = np.sum(points_per_shell)
    rs = (np.arange(nb_shells) + 1) / nb_shells # Relative shell radii
    shell_groups = [[i] for i in range(nb_shells)]
    shell_groups.append(range(nb_shells))
    alphas = np.ones(len(shell_groups))/2
    weights = ms.compute_weights(nb_shells, points_per_shell, shell_groups, alphas)
    points = ms.optimize(nb_shells, points_per_shell, weights, max_iter=1000)

    basename = output if output else '%02d_shells-%s' % (nb_shells, '-'.join(str(K_s) for K_s in points_per_shell))
    filename = '%s_samples.txt' % basename
    ms.write(points, nb_shells, points_per_shell, filename)

    # Visualization (can be commented out if matplotlib is not always desired)
    fig = plt.figure(figsize=(3.0 * nb_shells, 3.0))
    spacing = 0.05
    plt.subplots_adjust(left=spacing / nb_shells, right=1 - spacing / nb_shells, bottom=spacing, top=1 - spacing, wspace=2 * spacing / nb_shells)
    visu_points.draw_points_sphere(points, nb_shells, points_per_shell, rs, fig)
    plt.suptitle("Individual shells")
    plt.savefig("%s_shells.png" % basename, dpi=400)
    plt.close(fig)

    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=spacing / nb_shells, right=1 - spacing / nb_shells, bottom=spacing, top=1 - spacing, wspace=2 * spacing / nb_shells)
    visu_points.draw_points_reprojected(points, nb_shells, points_per_shell, rs, ax)
    plt.suptitle("Shells reprojected")
    plt.savefig("%s_shells-reprojected.png" % basename, dpi=400)
    plt.close(fig)

