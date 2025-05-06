import numpy as np
import os
import math
import argparse


def save_slice_timing_files(slice_timing_list, folder_path):
    """
    Saves the list of slice timing 2D arrays to text files.
    Each file corresponds to a volume.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created output directory: {folder_path}")
    for vol_idx, st_array in enumerate(slice_timing_list):
        if st_array is not None and st_array.shape[0] > 0:  # Only save if there are slices
            np.savetxt(os.path.join(folder_path, f"GE_{vol_idx}_slspec.txt"), st_array, fmt='%d', delimiter=' ')


def generate_slice_timing_for_gradient_cycling(
        mb_factor,
        gc_tr_block_size,
        num_volumes,
        num_total_slices):
    """
        Generates slice timing specifications for GE gradient cycling sequences.

        Args:
            mb_factor (int): Multiband factor.
            gc_tr_block_size (int): Gradient cycling TR block size (e.g., 2 for 2TR, 3 for 3TR).
                                    This is the number of volumes in one gradient cycle.
            num_volumes (int): Total number of volumes in the acquisition.
                               The first volume (index 0) is assumed to be a b0 not affected
                               by gradient cycling.
            num_total_slices (int): Total number of physical slices in the volume.
    """

    if num_total_slices % mb_factor != 0:
        raise ValueError(f"Total number of slices ({num_total_slices}) must be divisible by MB factor ({mb_factor}).")
    if gc_tr_block_size < 1:
        raise ValueError("Gradient cycling TR block size (gc_tr_block_size) must be at least 1.")
    if num_volumes < 1:
        raise ValueError("Number of volumes must be at least 1.")

    num_shots = int(num_total_slices // mb_factor)

    slice_timings = np.zeros((num_volumes, num_shots, mb_factor), dtype=int)

    # Step 1: Populate ALL volumes with the Volume 0 acquisition pattern
    # (Interleaved acquisition of all slice packets)
    l = 0
    for i in range(1, num_shots+1, 2):
        for m in range(mb_factor):
            slice_timings[:, l, m] = i - 1 + m * num_shots
        l = l+1

    for i in range(2, num_shots+1, 2):
        for m in range(mb_factor):
            slice_timings[:, l, m] = i - 1 + m * num_shots
        l = l+1

    # Step 2: For cycled volumes (vol 1 to N-1)
    for vol in range(1, num_volumes-1, gc_tr_block_size):
        curr_tr_block = 0
        j = {}
        for k in range(0, gc_tr_block_size):
            for i in range(1, num_shots+1, 2):
                if curr_tr_block not in j.keys():
                    j[curr_tr_block] = 0
                for m in range(mb_factor):
                    slice_timings[vol + curr_tr_block, j[curr_tr_block], m] = i - 1 + m * num_shots
                j[curr_tr_block] += 1

                if curr_tr_block < (gc_tr_block_size-1):
                    curr_tr_block += 1
                else:
                    curr_tr_block = 0

            for i in range(2, num_shots+1, 2):
                if curr_tr_block not in j.keys():
                    j[curr_tr_block] = 0
                for m in range(mb_factor):
                    slice_timings[vol + curr_tr_block, j[curr_tr_block], m] = i - 1 + m * num_shots

                j[curr_tr_block] += 1
                if curr_tr_block < (gc_tr_block_size-1):
                    curr_tr_block += 1
                else:
                    curr_tr_block = 0

    return slice_timings

def main():
    parser = argparse.ArgumentParser(
        description="Generate slice timing files (slspec.txt) for GE gradient cycling sequences.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # 3TR-like sequence
  python %(prog)s --mb 3 --gc_tr 3 --nvols 169 --nslice 69 --output_dir sliceTiming/st_3TR

  # 2TR-like sequence
  python %(prog)s --mb 3 --gc_tr 2 --nvols 169 --nslice 69 --output_dir sliceTiming/st_2TR

  # "AllTR" like sequence (each cycled volume gets one packet if available)
  python %(prog)s --mb 3 --gc_tr 168 --nvols 169 --nslice 69 --output_dir sliceTiming/st_ALLTR
"""
    )
    parser.add_argument("--mb", type=int, required=True, help="Multiband factor.")
    parser.add_argument("--gc_tr", type=int, required=True,
                        help="Gradient Cycling TR block size (number of volumes in one gradient cycle).")
    parser.add_argument("--nvols", type=int, required=True,
                        help="Total number of volumes in the acquisition (includes non-cycled vol 0).")
    parser.add_argument("--nslice", type=int, required=True, help="Total number of physical slices in the volume.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the generated slspec.txt files.")

    args = parser.parse_args()

    print(
        f"--- Generating Slice Timings for: MB={args.mb}, GC_TR={args.gc_tr}, NVol={args.nvols}, NSlice={args.nslice} ---")

    try:
        slice_timings_list = generate_slice_timing_for_gradient_cycling(
            mb_factor=args.mb,
            gc_tr_block_size=args.gc_tr,
            num_volumes=args.nvols,
            num_total_slices=args.nslice
        )

        save_slice_timing_files(slice_timings_list, args.output_dir)
        print(f"Slice timing files saved to: {args.output_dir}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()