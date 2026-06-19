#!/usr/bin/env python
# Run a real vLLM CAA hook smoke test and record baseline, steered, and restored outputs.

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROMPTS = [
    "The capital of France is",
    "A careful answer to the user should",
]


def load_hook_module(repo_root):
    steer_path = str(repo_root / "steer")
    if steer_path not in sys.path:
        sys.path.insert(0, steer_path)
    return importlib.import_module("vllm_caa_hooks")


def maybe_start_gpu_monitor(output_path, interval_seconds):
    if interval_seconds <= 0:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
        "--format=csv",
        f"--loop-ms={int(interval_seconds * 1000)}",
    ]
    try:
        output_file = output_path.open("w", encoding="utf-8")
        return subprocess.Popen(command, stdout=output_file, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return None


def stop_gpu_monitor(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--multiplier", type=float, default=0.0)
    parser.add_argument("--vector-value", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--monitor-output")
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0)
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    parser.add_argument("--disable-enforce-eager", dest="enforce_eager", action="store_false")
    parser.set_defaults(enforce_eager=True)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def request_texts(outputs):
    return [output.outputs[0].text for output in outputs]


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output)
    monitor_path = Path(args.monitor_output) if args.monitor_output else output_path.with_suffix(".gpu.csv")
    hook_module = load_hook_module(repo_root)

    import torch
    import vllm
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    monitor = maybe_start_gpu_monitor(monitor_path, args.monitor_interval_seconds)
    started_at = time.time()
    llm = None
    try:
        model_config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            seed=args.seed,
            async_scheduling=False,
        )
        prompts = args.prompts or DEFAULT_PROMPTS
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

        baseline = llm.generate(prompts, sampling_params, use_tqdm=False)
        hidden_size = int(model_config.hidden_size)
        vector = torch.full((hidden_size,), args.vector_value, dtype=torch.float32)

        install_results = hook_module.install_caa_with_vllm_rpc(
            llm,
            layer_vectors={args.layer: vector},
            multipliers={args.layer: args.multiplier},
        )
        steered = llm.generate(prompts, sampling_params, use_tqdm=False)
        hook_stats_before_clear = hook_module.get_caa_hook_stats_with_vllm_rpc(llm)
        clear_results = hook_module.clear_caa_with_vllm_rpc(llm)
        hook_stats_after_clear = hook_module.get_caa_hook_stats_with_vllm_rpc(llm)
        restored = llm.generate(prompts, sampling_params, use_tqdm=False)

        baseline_text = request_texts(baseline)
        steered_text = request_texts(steered)
        restored_text = request_texts(restored)
        result = {
            "metadata": {
                "model": args.model,
                "vllm_version": vllm.__version__,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "tensor_parallel_size": args.tensor_parallel_size,
                "layer": args.layer,
                "multiplier": args.multiplier,
                "vector_value": args.vector_value,
                "sampling": {
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
                "model_config": {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
                    "num_attention_heads": getattr(model_config, "num_attention_heads", None),
                    "num_key_value_heads": getattr(model_config, "num_key_value_heads", None),
                },
                "install_results": install_results,
                "clear_results": clear_results,
                "hook_stats_before_clear": hook_stats_before_clear,
                "hook_stats_after_clear": hook_stats_after_clear,
                "elapsed_seconds": time.time() - started_at,
                "gpu_monitor_output": str(monitor_path),
                "enable_prefix_caching": args.enable_prefix_caching,
                "async_scheduling": False,
            },
            "prompts": prompts,
            "baseline": baseline_text,
            "steered": steered_text,
            "restored": restored_text,
            "comparisons": {
                "baseline_equals_restored": baseline_text == restored_text,
                "baseline_equals_steered": baseline_text == steered_text,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        if not result["comparisons"]["baseline_equals_restored"]:
            raise AssertionError("baseline and restored outputs differ after clearing CAA hooks")
    finally:
        if llm is not None:
            del llm
        stop_gpu_monitor(monitor)

if __name__ == "__main__":
    main()
