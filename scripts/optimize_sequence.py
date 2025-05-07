import argparse
import sys
import os

# Add the package directory to the Python path if running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import core, optimizers

def main():
    parser = argparse.ArgumentParser(description="Optimize dMRI Gradient Sequence for Thermal Efficiency.")
    parser.add_argument("input_tensor", help="Path to the input GE tensor.dat file.")
    parser.add_argument("output_tensor", help="Path to save the optimized GE tensor.dat file.")
    parser.add_argument("-m", "--method", type=str, default="iterated_local_search",
                        help=f"Optimization method to use (default: iterated_local_search). Available: {[m for m in dir(optimizers) if not m.startswith('_')]}")
    parser.add_argument("-g", "--group_size", type=int, default=3, choices=[2, 3],
                        help="TR group size for optimization (default: 3).")
    parser.add_argument("--weight_total_power", type=float, default=0.05,
                        help="Weight for the total power term in cost function (default: 0.05).")
    parser.add_argument("--weight_adjacent_group", type=float, default=0.1,
                        help="Weight for the adjacent group term in cost function (default: 0.1).")
    parser.add_argument("--n_dirs", type=int, default=None,
                        help="Specify the number of directions for the block to optimize within the tensor file. "
                             "If not provided (default: None), the script will optimize the block "
                             "with the largest number of directions found in the file. "
                             "If a number is provided, the first block found with exactly this "
                             "many directions will be targeted for optimization.")
    parser.add_argument("--n_iter", type=int, default=10000,
                        help="Number of iterations for the optimizer (default: 10000) (should be lower for ILS).")
    parser.add_argument("--n_permute", type=int, default=6,
                        help="Number of vectors to permute in each smart brute force step (default: 6).")
    parser.add_argument("--ils_depth", type=int, default=100,
                        help="Local search depth (max swaps without improvement) for ILS (default: 100).")
    parser.add_argument("--ils_perturb", type=int, default=6,
                        help="Perturbation strength (number of swaps) for ILS (default: 6).")

    args = parser.parse_args()

    if not os.path.exists(args.input_tensor):
        print(f"Error: Input tensor file not found: {args.input_tensor}")
        sys.exit(1)

    # Prepare optimizer arguments
    optimizer_kwargs = {
        'n_iter': args.n_iter,
    }
    if args.method == 'smart_brute_force':
        if hasattr(args, 'n_permute'):
            optimizer_kwargs['N_to_permute'] = args.n_permute
    elif args.method == 'iterated_local_search':
        if hasattr(args, 'ils_depth'):
            optimizer_kwargs['local_search_depth'] = args.ils_depth
        if hasattr(args, 'ils_perturb'):
            optimizer_kwargs['perturbation_strength'] = args.ils_perturb

    # --- Run Optimization ---
    try:
        core.optimize_gradient_sequence(
            input_tensor_path=args.input_tensor,
            output_tensor_path=args.output_tensor,
            method=args.method,
            group_size=args.group_size,
            n_dirs_selected=args.n_dirs,
            weight_total_power=args.weight_total_power,
            weight_adjacent_group=args.weight_adjacent_group,
            optimizer_kwargs=optimizer_kwargs
        )
        print("\nOptimization finished successfully!")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()