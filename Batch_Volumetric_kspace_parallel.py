"""
Parallel batch processing script for Volumetric k-space reconstruction.
Runs multiple reconstruction tasks concurrently instead of sequentially.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime


def get_deepest_directories(root_path):
    """Find all directories at maximum depth."""
    all_dirs = []
    max_depth = 0

    for root, dirs, files in os.walk(root_path):
        depth = len(Path(root).parts)
        if depth > max_depth:
            max_depth = depth
        all_dirs.append((root, depth))

    deepest = [path for path, depth in all_dirs if depth == max_depth]
    return sorted(deepest)


def get_mask_files(mask_folder):
    """Get all BMP mask files."""
    mask_files = list(Path(mask_folder).glob("*.bmp"))
    return sorted([str(f) for f in mask_files])


def run_reconstruction_task(task_params):
    """
    Run a single reconstruction task.

    Args:
        task_params: dict with keys:
            - input_path: path to dataset
            - mask_path: path to mask BMP
            - output_path: where to save results
            - repo_path: path to repo
            - target_size, wavelet_name, wavelet_level, reg_param, iter_num, tolerance
            - nifti_folder: path for NIFTI outputs
    """
    try:
        input_path = task_params["input_path"]
        mask_path = task_params["mask_path"]
        output_path = task_params["output_path"]
        repo_path = task_params["repo_path"]
        target_size = task_params["target_size"]
        wavelet_name = task_params["wavelet_name"]
        wavelet_level = task_params["wavelet_level"]
        reg_param = task_params["reg_param"]
        iter_num = task_params["iter_num"]
        tolerance = task_params["tolerance"]
        nifti_folder = task_params["nifti_folder"]

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Build command - use venv python directly
        script_path = os.path.join(repo_path, "Volumetric", "Volumetric_kspace_final.py")
        venv_python = os.path.join(repo_path, ".venv", "Scripts", "python.exe")

        # Use venv python if available, otherwise fallback to system python
        python_exe = venv_python if os.path.exists(venv_python) else "python"

        cmd = [
            python_exe,
            script_path,
            input_path,
            str(target_size),
            wavelet_name,
            str(wavelet_level),
            str(reg_param),
            str(iter_num),
            str(tolerance),
            output_path,
            mask_path,
            nifti_folder
        ]

        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        dataset_name = os.path.basename(input_path)
        mask_name = Path(mask_path).stem

        if result.returncode == 0:
            return {
                "status": "success",
                "dataset": dataset_name,
                "mask": mask_name,
                "message": f"[success] {dataset_name} + {mask_name}"
            }
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return {
                "status": "error",
                "dataset": dataset_name,
                "mask": mask_name,
                "message": f"[error] {dataset_name} + {mask_name}:\n{error_msg}"
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "dataset": task_params.get("input_path", "unknown"),
            "mask": task_params.get("mask_path", "unknown"),
            "message": f"[timeout] {task_params.get('input_path', 'unknown')}"
        }
    except Exception as e:
        return {
            "status": "error",
            "dataset": task_params.get("input_path", "unknown"),
            "mask": task_params.get("mask_path", "unknown"),
            "message": f"[error] Exception: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch processing for Volumetric k-space reconstruction"
    )
    parser.add_argument("--input", required=True, help="Root input data path")
    parser.add_argument("--masks", required=True, help="Folder containing BMP masks")
    parser.add_argument("--nifti", default=None, help="NIFTI output folder (optional)")
    parser.add_argument("--target-size", type=int, default=256, help="Target image size")
    parser.add_argument("--wavelet", default="haar", help="Wavelet name")
    parser.add_argument("--wavelet-level", type=int, default=7, help="Wavelet decomposition level")
    parser.add_argument("--reg-param", type=float, default=5, help="Regularization parameter")
    parser.add_argument("--iterations", type=int, default=10, help="Iteration number")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Convergence tolerance")
    parser.add_argument("--workers", type=int, default=None,
                       help=f"Number of parallel workers (default: {cpu_count()})")
    parser.add_argument("--repo", default=r"C:\Users\osman\Documents\GitHub\CompressedSensingforMRI",
                       help="Path to CompressedSensingforMRI repo")

    args = parser.parse_args()

    # Determine number of workers
    num_workers = args.workers or max(1, cpu_count() - 1)

    # Setup paths
    input_root = args.input
    mask_folder = args.masks
    nifti_folder = args.nifti
    repo_path = args.repo

    # Create results directory
    root_parent = os.path.dirname(input_root)
    root_leaf = os.path.basename(input_root)
    results_leaf = root_leaf + " Results Volumetric Final"
    results_path = os.path.join(root_parent, results_leaf)
    os.makedirs(results_path, exist_ok=True)

    # Create NIFTI output directory if not specified
    if nifti_folder is None:
        nifti_folder = os.path.join(root_parent, root_leaf + " Nifti Outputs")
    os.makedirs(nifti_folder, exist_ok=True)

    # Get datasets and masks
    deepest_dirs = get_deepest_directories(input_root)
    mask_files = get_mask_files(mask_folder)

    print(f"\n{'='*70}")
    print(f"Parallel Volumetric k-space Batch Processing")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Found {len(deepest_dirs)} dataset(s)")
    print(f"Found {len(mask_files)} mask(s)")
    print(f"Total tasks: {len(deepest_dirs) * len(mask_files)}")
    print(f"Worker processes: {num_workers}")
    print(f"Results directory: {results_path}")
    print(f"{'='*70}\n")

    if not mask_files:
        print("ERROR: No BMP files found in mask folder!")
        sys.exit(1)

    # Build task list
    tasks = []
    for mask_path in mask_files:
        mask_name = Path(mask_path).stem
        for dataset_path in deepest_dirs:
            dataset_name = os.path.basename(dataset_path)
            output_path = os.path.join(results_path, f"{dataset_name} _ {mask_name} Results")

            task = {
                "input_path": dataset_path,
                "mask_path": mask_path,
                "output_path": output_path,
                "repo_path": repo_path,
                "target_size": args.target_size,
                "wavelet_name": args.wavelet,
                "wavelet_level": args.wavelet_level,
                "reg_param": args.reg_param,
                "iter_num": args.iterations,
                "tolerance": args.tolerance,
                "nifti_folder": nifti_folder,
            }
            tasks.append(task)

    # Run tasks in parallel
    print("Starting parallel processing...\n")
    try:
        with Pool(num_workers) as pool:
            results = pool.imap_unordered(run_reconstruction_task, tasks, chunksize=1)

            success_count = 0
            error_count = 0

            for i, result in enumerate(results, 1):
                print(f"[{i:3d}/{len(tasks)}] {result['message']}")
                if result["status"] == "success":
                    success_count += 1
                else:
                    error_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        sys.exit(1)

    # Summary
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {success_count}/{len(tasks)}")
    print(f"Failed: {error_count}/{len(tasks)}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
