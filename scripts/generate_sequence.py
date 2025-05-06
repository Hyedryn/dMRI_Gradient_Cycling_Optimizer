import argparse
import os
import sys

# Add the package directory to the Python path if running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import io

def main():
    parser = argparse.ArgumentParser(description="Generate GE tensor.dat file using qspace sampling.")
    parser.add_argument("output_prefix", help="Prefix for output files (e.g., 'output/my_scheme'). _samples.txt and _tensor.dat will be appended.")
    parser.add_argument("-b", "--bvalues", required=True, nargs='+', type=float, help="List of b-values (e.g., 1000 2000 3000).")
    parser.add_argument("-n", "--ndirs", required=True, nargs='+', type=int, help="List of number of directions per shell (must match bvalues order).")
    parser.add_argument("-b0", "--b0count", type=int, default=8, help="Number of b=0 volumes to include (default: 8).")
    parser.add_argument("--b0spacing", type=int, default=14, help="Insert a b=0 volume every N DWI volumes (0 for no interleaving, default: 14).")

    args = parser.parse_args()

    if len(args.bvalues) != len(args.ndirs):
        print("Error: Number of b-values must match number of direction counts.")
        sys.exit(1)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("Generating sequence...")
    print(f"  b-values: {args.bvalues}")
    print(f"  Dirs/shell: {args.ndirs}")
    print(f"  b0 count: {args.b0count}")
    print(f"  b0 spacing: {args.b0spacing}")

    try:
        tensor_dat_path = io.format_ge_tensor_for_sequence(
            bvalues=args.bvalues,
            vectors_per_shell=args.ndirs,
            b0_count=args.b0count,
            output_prefix=args.output_prefix,
            b0_spacing=args.b0spacing
        )
        print(f"\nSuccessfully generated tensor file: {tensor_dat_path}")
    except ImportError as e:
         print(f"\nError: {e}. Make sure 'qspace' and 'matplotlib' are installed.")
         sys.exit(1)
    except Exception as e:
        print(f"\nError during sequence generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()