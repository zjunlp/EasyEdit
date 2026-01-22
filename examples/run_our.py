#!/usr/bin/env python3
"""
ACL 2026 Experiment Runner

This script runs steering vector generation and application experiments
for different datasets (axbench, psychopathy, powerseeking), methods (caa, reps, sft, our),
and intervention methods (vector, lora, local_weight).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

VALID_METHODS: List[str] = ["caa", "reps", "sft", "our"]
VALID_INTERVENTION_METHODS: List[str] = ["vector", "lora", "local_weight"]


def _dedupe_keep_order(items: List[tuple]) -> List[tuple]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _expand_method_intervention_pairs(method: str, intervention_method: str) -> List[tuple]:
    """
    Expand (method, intervention_method) with support for 'all'.
    - method='all' => iterate all methods
    - intervention_method='all' => iterate all interventions
    - both='all' => iterate all valid pairs
    Notes:
      - CAA only supports vector; unsupported pairs are skipped (not forced).
    """
    methods = VALID_METHODS if method == "all" else [method]
    interventions = VALID_INTERVENTION_METHODS if intervention_method == "all" else [intervention_method]

    pairs: List[tuple] = []
    for m in methods:
        for im in interventions:
            if m == "caa" and im != "vector":
                print(f"[SKIP] CAA does not support intervention_method={im}; skipping.")
                continue
            pairs.append((m, im))
    return _dedupe_keep_order(pairs)


def _abs_script_path(base_dir: str, script_name: str) -> str:
    script_path = os.path.join(base_dir, script_name)
    return script_path if os.path.isabs(script_path) else os.path.abspath(script_path)


def _resolve_generate_hparam(
    base_dir: str,
    dataset: str,
    model_name: str,
    method: str,
    intervention_method: str,
) -> Optional[str]:
    """
    Resolve the training hparam yaml path for a given (dataset, model, method, intervention).
    Some folders use legacy filenames (e.g. caa uses generate_caa.yaml, our uses generate_our_weight.yaml).
    """
    base = f"{base_dir}/hparams/Steer/acl_experiment/{dataset}/{model_name}/{method}"
    candidates: List[str] = [
        f"{base}/generate_{method}_{intervention_method}.yaml",
    ]
    if intervention_method == "local_weight":
        candidates.append(f"{base}/generate_{method}_weight.yaml")
    if method == "caa":
        candidates.append(f"{base}/generate_caa.yaml")

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _resolve_apply_hparam(
    base_dir: str,
    dataset: str,
    model_name: str,
    method: str,
) -> Optional[str]:
    base = f"{base_dir}/hparams/Steer/acl_experiment/{dataset}/{model_name}/{method}"
    candidates: List[str] = [
        f"{base}/apply_{method}.yaml",
        f"{base}/apply.yaml",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _vector_load_subdir(method: str, intervention_method: str) -> str:
    # NOTE: for "our" configs, alg_name may still be "reps" (so reuse reps_* layout).
    if method == "caa":
        return "caa_vector"
    if method == "our":
        return f"reps_{intervention_method}"
    return f"{method}_{intervention_method}"


def run_vector_generation(
    dataset: str,
    method: str,
    model_name: str,
    intervention_method: str,
    device: str = "cuda:0",
    base_dir: str = ".",
    dry_run: bool = False,
):
    """
    Run vector generation.
    - dataset=axbench: use axbench_experiment.py (AxBench-only pipeline)
    - otherwise: use vectors_generate.py (generic pipeline)
    
    Args:
        dataset: Dataset name (axbench, psychopathy, powerseeking)
        method: Method name (caa, reps, sft, our)
        model_name: Model name (e.g., gemma-2-9b-it)
        intervention_method: Intervention method (vector, lora, local_weight)
        device: Device to use (e.g., cuda:0)
        base_dir: Base directory for the project
        dry_run: If True, only print the command without executing
    """
    leaf_dir = _vector_load_subdir(method, intervention_method)
    # Set up paths
    log_path = f"{base_dir}/vectors/{model_name}/{method}/{dataset}/{leaf_dir}/train.log"
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    
    train_hparam = _resolve_generate_hparam(
        base_dir=base_dir,
        dataset=dataset,
        model_name=model_name,
        method=method,
        intervention_method=intervention_method,
    )
    
    if not train_hparam:
        expected = f"{base_dir}/hparams/Steer/acl_experiment/{dataset}/{model_name}/{method}/..."
        print(f"[WARNING] Hparam file not found under: {expected}")
        print(f"[SKIP] Skipping {method} with {intervention_method} for {dataset}")
        return True
    
    if dataset == "axbench":
        axbench_experiment_path = _abs_script_path(base_dir, "axbench_experiment.py")
        cmd = [
            sys.executable,
            axbench_experiment_path,
            f"device={device}",
            f"+axbench_output_dir_name={method}_{intervention_method}",
            f"steer_train_hparam_paths=[{train_hparam}]",
            f"steer_vector_output_dirs=[vectors/{model_name}/{method}]",
        ]
    else:
        vectors_generate_path = _abs_script_path(base_dir, "vectors_generate.py")
        cmd = [
            sys.executable,
            vectors_generate_path,
            f"device={device}",
            f"model_name_or_path=./models/{model_name}",
            f"steer_train_dataset=[{dataset}]",
            f"steer_train_hparam_paths=[{train_hparam}]",
            # Keep method-specific isolation to avoid collisions (e.g., our vs reps)
            f"steer_vector_output_dirs=[vectors/{model_name}/{method}]",
        ]
    
    print(f"[INFO] Running vector generation: {method} with {intervention_method} for {dataset}")
    print(f"[INFO] Command: {' '.join(cmd)}")

    if dry_run:
        print(f"[DRY_RUN] Log would be written to: {log_path}")
        return True
    
    # Run command and redirect output to log file
    with open(log_path, 'w') as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=base_dir)
    
    if result.returncode == 0:
        print(f"[SUCCESS] Vector generation completed. Log: {log_path}")
        return True
    else:
        print(f"[ERROR] Vector generation failed. Check log: {log_path}")
        return False


def run_vector_application(
    dataset: str,
    method: str,
    model_name: str,
    intervention_method: str,
    multipliers: list,
    device: str = "cuda:0",
    base_dir: str = ".",
    dry_run: bool = False,
):
    """
    Run vector application.
    - dataset=axbench: use axbench_infer.py (AxBench-only pipeline)
    - otherwise: use vectors_apply.py (generic pipeline)
    
    Args:
        dataset: Dataset name (axbench, psychopathy, powerseeking)
        method: Method name (caa, reps, sft, our)
        model_name: Model name (e.g., gemma-2-9b-it)
        intervention_method: Intervention method (vector, lora, local_weight)
        multipliers: List of multiplier values (e.g., [1.0, 2.0])
        device: Device to use (e.g., cuda:0)
        base_dir: Base directory for the project
        dry_run: If True, only print the command without executing
    """
    results = []
    
    for m in multipliers:
        log_path = f"{base_dir}/generation/{model_name}/{method}/{dataset}/{method}_{intervention_method}/logs/{method}_{intervention_method}_m{m}.log"
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)
        
        apply_hparam = _resolve_apply_hparam(
            base_dir=base_dir,
            dataset=dataset,
            model_name=model_name,
            method=method,
        )
        
        if not apply_hparam:
            expected = f"{base_dir}/hparams/Steer/acl_experiment/{dataset}/{model_name}/{method}/..."
            print(f"[WARNING] Apply hparam file not found under: {expected}")
            results.append(True)
            continue
        
        if dataset == "axbench":
            axbench_infer_path = _abs_script_path(base_dir, "axbench_infer.py")
            cmd = [
                sys.executable,
                axbench_infer_path,
                f"device={device}",
                f"+model_name={model_name}",
                f"+method={method}",
                f"apply_steer_hparam_paths=[{apply_hparam}]",
                f"+multipliers=[{m}]",
                f"vector_name={method}_{intervention_method}",
                f"generation_output_dir=generation/{model_name}/{method}/{dataset}/{method}_{intervention_method}",
                f"+intervention_method={intervention_method}",
            ]
        else:
            vectors_apply_path = _abs_script_path(base_dir, "vectors_apply.py")
            load_subdir = _vector_load_subdir(method, intervention_method)
            steer_vector_load_dir = f"vectors/{model_name}/{method}/{dataset}/{load_subdir}"
            generation_output_dir = f"generation/{model_name}/{method}/{dataset}/{method}_{intervention_method}/m{m}"
            cmd = [
                sys.executable,
                vectors_apply_path,
                f"device={device}",
                f"model_name_or_path=./models/{model_name}",
                f"apply_steer_hparam_paths=[{apply_hparam}]",
                f"steer_vector_load_dir=[{steer_vector_load_dir}]",
                f"generation_data=[{dataset}]",
                f"generation_output_dir={generation_output_dir}",
                f"+multipliers=[{m}]",
                f"+intervention_method={intervention_method}",
            ]
        
        print(f"[INFO] Running vector application: {method} with {intervention_method}, multiplier={m} for {dataset}")
        print(f"[INFO] Command: {' '.join(cmd)}")

        if dry_run:
            print(f"[DRY_RUN] Log would be written to: {log_path}")
            results.append(True)
            continue
        
        # Run command and redirect output to log file
        with open(log_path, 'w') as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=base_dir)
        
        if result.returncode == 0:
            print(f"[SUCCESS] Vector application completed. Log: {log_path}")
            results.append(True)
        else:
            print(f"[ERROR] Vector application failed. Check log: {log_path}")
            results.append(False)
    
    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run ACL 2026 experiments for steering vector generation and application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate vectors for axbench dataset using reps method with vector intervention
  python run_ACL2026.py --dataset axbench --method reps --model_name gemma-2-9b-it --intervention_method vector --mode generate

  # Apply vectors for axbench dataset using sft method with lora intervention
  python run_ACL2026.py --dataset axbench --method sft --model_name gemma-2-9b-it --intervention_method lora --mode apply --multipliers 1.0 2.0

  # Run both generation and application
  python run_ACL2026.py --dataset axbench --method reps --model_name gemma-2-9b-it --intervention_method vector --mode both
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True, type=str,
                       choices=['axbench', 'psychopathy', 'powerseeking'],
                       help='Dataset name')
    parser.add_argument('--method', required=True, type=str,
                       choices=VALID_METHODS + ['all'],
                       help='Method name (or all)')
    parser.add_argument('--model_name', required=True, type=str,
                       help='Model name (e.g., gemma-2-9b-it, qwen2.5-7b-it)')
    parser.add_argument('--intervention_method', required=True, type=str,
                       choices=VALID_INTERVENTION_METHODS + ['all'],
                       help='Intervention method (or all)')
    
    # Optional arguments
    parser.add_argument('--mode', default='both', type=str,
                       choices=['generate', 'apply', 'both'],
                       help='Mode: generate vectors, apply vectors, or both (default: both)')
    parser.add_argument('--device', default='cuda:0', type=str,
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--multipliers', nargs='+', type=float, default=[1.0],
                       help='Multiplier values for vector application (default: [1.0])')
    parser.add_argument('--base_dir', default='.', type=str,
                       help='Base directory for the project (default: current directory)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only print commands (do not execute). Useful to verify dataset/method routing.')
    
    args = parser.parse_args()
    
    pairs = _expand_method_intervention_pairs(args.method, args.intervention_method)
    if not pairs:
        print("[WARNING] No valid (method, intervention_method) pairs to run.")
        return 1
    
    print("=" * 80)
    print("ACL 2026 Experiment Runner")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Model: {args.model_name}")
    print(f"Intervention Method: {args.intervention_method}")
    print(f"Expanded pairs: {pairs}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    if args.mode in ['apply', 'both']:
        print(f"Multipliers: {args.multipliers}")
    print("=" * 80)
    
    success = True
    
    # Run vector generation
    if args.mode in ['generate', 'both']:
        print("\n[PHASE 1] Vector Generation")
        print("-" * 80)
        for method, intervention_method in pairs:
            gen_success = run_vector_generation(
                dataset=args.dataset,
                method=method,
                model_name=args.model_name,
                intervention_method=intervention_method,
                device=args.device,
                base_dir=args.base_dir,
                dry_run=args.dry_run,
            )
            success = success and gen_success
    
    # Run vector application
    if args.mode in ['apply', 'both']:
        print("\n[PHASE 2] Vector Application")
        print("-" * 80)
        for method, intervention_method in pairs:
            app_success = run_vector_application(
                dataset=args.dataset,
                method=method,
                model_name=args.model_name,
                intervention_method=intervention_method,
                multipliers=args.multipliers,
                device=args.device,
                base_dir=args.base_dir,
                dry_run=args.dry_run,
            )
            success = success and app_success
    
    # Summary
    print("\n" + "=" * 80)
    if success:
        print("[SUCCESS] All experiments completed successfully!")
    else:
        print("[WARNING] Some experiments failed. Please check the logs above.")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
