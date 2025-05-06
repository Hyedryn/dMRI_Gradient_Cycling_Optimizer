# dMRI Gradient Cycling Optimizer for thermal load management on GE scanners

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Optimize the order of diffusion gradient vectors in GE tensor.dat files to improve thermal efficiency for gradient cycling sequences (2TR or 3TR).

## Problem

Gradient cycling is a technique used in diffusion MRI (dMRI) acquisitions, particularly on GE scanners (e.g., Signa Premier), to manage the thermal load on gradient coils during long, demanding sequences (like high b-value multi-shell scans). This involves cycling through different gradient directions/strengths across slices/TRs.

However, the *order* in which gradients are applied within a cycling block (e.g., a 2TR or 3TR block) significantly impacts the instantaneous and cumulative thermal load. Applying gradients with similar orientations or magnitudes close together in time can lead to higher peak loads. Optimizing the sequence order aims to distribute the load more evenly, potentially allowing for shorter TRs and more efficient acquisitions.

## Solution

This repository provides tools to reorder the gradient vectors within a standard GE `tensor.dat` file based on minimizing a cost function that reflects the interaction between gradients within a defined TR group (typically 2 or 3).

The primary cost function minimized (`eval_ge_cycling_cost`) calculates the sum of absolute dot products between pairs of gradient vectors within each group. Minimizing this cost aims to maximize the angular separation between subsequent gradient applications within a group, thus reducing coherent gradient effects contributing to heating.

Several optimization algorithms are provided, with `iterated_local_search` being the default method.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hyedryn/dMRI_Gradient_Cycling_Optimizer.git
    cd dMRI_Gradient_Cycling_Optimizer
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **(Optional) Install `qspace` for Sequence Generation:**
    The script `scripts/generate_sequence.py` requires the `qspace` library for generating the initial diffusion sampling scheme. `qspace` is not available on PyPI (pip). You need to install it manually from its GitHub repository:
    ```bash
    # Clone the qspace repository 
    git clone https://github.com/ecaruyer/qspace.git ../qspace_source

    # Navigate into the cloned directory
    cd ../qspace_source

    # Install qspace
    python setup.py install --user

    # Navigate back to the optimizer directory
    cd ../dMRI_Gradient_Cycling_Optimizer
    ```
    *Note: If you already have a GE `tensor.dat` file, you do not need to install `qspace`.*

## Usage

### Command Line Interface (Recommended)

The primary way to use the tool is via the `optimize_sequence.py` script:

```bash
python scripts/optimize_sequence.py <input_tensor.dat> <output_tensor.dat> [options]
```

**Arguments:**

*   `input_tensor.dat`: Path to the original GE tensor file.
*   `output_tensor.dat`: Path where the optimized tensor file will be saved.

**Options:**

*   `-m METHOD`, `--method METHOD`: Optimization algorithm (default: `iterated_local_search`). See available methods below.
*   `-g GROUP_SIZE`, `--group_size GROUP_SIZE`: TR group size (2 or 3, default: 3).
*   `--n_iter N_ITER`: Number of iterations for the optimizer (default: 10000).
*   `--n_permute N_PERMUTE`: Number of vectors to permute in each `smart_brute_force` step (default: 6).
*   `--ils_depth ILS_DEPTH`: Local search depth (max swaps without improvement) for ILS (default: 50).
*   `--ils_perturb ILS_PERTURB`: Perturbation strength (number of swaps) for ILS (default: 5).

**Example:**

```bash
python scripts/optimize_sequence.py data/my_original_tensor.dat output/my_optimized_tensor_3TR.dat -g 3 --n_iter 50000
```

### Python Library Usage

You can also import and use the core optimization function in your Python scripts:

```python
from src import core

input_file = "data/my_original_tensor.dat"
output_file = "output/my_optimized_tensor_2TR.dat"
group = 2
method_name = 'iterated_local_search'
opt_kwargs = {'n_iter': 20000, 'ils_depth': 50, 'ils_perturb': 5}

try:
    optimized_vectors = core.optimize_gradient_sequence(
        input_tensor_path=input_file,
        output_tensor_path=output_file,
        method=method_name,
        group_size=group,
        optimizer_kwargs=opt_kwargs
    )
    print("Optimization successful!")
except Exception as e:
    print(f"Error: {e}")

```

## Input Format

The tool expects input files in the GE `tensor.dat` format. This typically includes:
*   Header lines starting with `#`.
*   A line indicating the number of directions in the subsequent block.
*   Lines containing the X, Y, Z components of each gradient vector for that block.
The optimizer works on the biggest block of directions found.

## Optimizers Available

*   `iterated_local_search` (Recommended): Uses iterated local search, applying perturbations to escape local minima found by random swaps.
*   `smart_brute_force`: Iteratively permutes subsets of vectors, focusing on the group with the highest cost function value.

## (Optional) Sequence Generation

A script `scripts/generate_sequence.py` is provided to generate a `tensor.dat` file using the `qspace` library. This requires `qspace` and `matplotlib` to be installed.

```bash
python scripts/generate_sequence.py output/my_new_scheme \
    --bvalues 1000 2000 3000 \
    --ndirs 64 64 64 \
    --b0count 12 \
    --b0spacing 14
```
This will create `output/my_new_scheme_samples.txt` and `output/my_new_scheme_tensor.dat`.

## (Optional) Slice Timing Generation for Custom Eddy

The script `scripts/generate_slice_timing.py` generates slice timing files (e.g., `GE_X_slspec.txt`) required by some custom versions of FSL's Eddy tool, particularly those adapted for GE gradient cycling sequences.

The script is run from the command line with arguments:
```bash
python scripts/generate_slice_timing.py \
    --mb <multiband_factor> \
    --gc_tr <gc_block_size> \
    --nvols <total_volumes> \
    --nslice <total_slices> \
    --output_dir <path_to_output_directory>
```
**Arguments:**

*   `--mb MB`: Multiband factor (integer).
*   `--gc_tr GC_TR`: Gradient Cycling TR block size (number of volumes in one gradient cycle, e.g., 2 or 3).
*   `--nvols NVOLS`: Total number of volumes in the acquisition (includes the initial non-cycled volume 0).
*   `--nslice NSLICE`: Total number of physical slices in the imaging volume.
*   `--output_dir OUTPUT_DIR`: Directory where the generated `GE_X_slspec.txt` files will be saved.

**Example:**

```bash
python scripts/generate_slice_timing.py \
    --mb 3 \
    --gc_tr 3 \
    --nvols 169 \
    --nslice 69 \
    --output_dir sliceTiming/my_3TR_slspec
```

## Citation

If you use this software in your research, please cite the relevant thesis/publication:

*   Dessain, Q. (2025). *Brain biomarkers from diffusion MRI : correction of motion artifacts, acceleration of microstructure estimation, and disease classification with transformers.* [UCLouvain].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
