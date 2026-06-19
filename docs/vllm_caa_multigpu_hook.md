<!--
Intent: Document the small PR surface for multi-GPU vLLM CAA steering hooks.
Updated: 2026-06-19
Branch: pr/vllm-caa-multigpu-hook
-->

# vLLM CAA Multi-GPU Hook

## Purpose

This PR adds a lightweight CAA activation-add hook for vLLM models, including tensor-parallel workers. The hook is intended for applying already-computed CAA vectors during vLLM inference.

The PR scope is deliberately small:

- Install a CAA vector on selected decoder layers inside vLLM worker-local models.
- Clear installed CAA vectors after generation.
- Read hook call/configuration stats from each worker.
- Provide a small GPU smoke script that checks baseline, steered, and restored generation.

Effect-consistency experiments are validation evidence only. They are not part of the runtime API.

## Runtime API

The implementation lives in `steer/vllm_caa_hooks.py`.

Use these helpers for tensor-parallel vLLM engines:

```python
from steer.vllm_caa_hooks import (
    clear_caa_with_vllm_rpc,
    get_caa_hook_stats_with_vllm_rpc,
    install_caa_with_vllm_rpc,
)

install_caa_with_vllm_rpc(
    llm,
    layer_vectors={12: caa_vector},
    multipliers={12: 4.0},
)

stats = get_caa_hook_stats_with_vllm_rpc(llm)

clear_caa_with_vllm_rpc(llm)
```

For single-process vLLM-style models, the lower-level functions are:

```python
from steer.vllm_caa_hooks import clear_vllm_caa_hooks, install_vllm_caa_hooks

install_vllm_caa_hooks(model, layer_vectors={12: caa_vector}, multipliers={12: 4.0})
clear_vllm_caa_hooks(model)
```

## Validation

Run the hook-only unit tests:

```bash
pytest -q tests/test_vllm_caa_hooks.py
```

Run syntax checks for the PR surface:

```bash
python -m compileall steer/vllm_caa_hooks.py examples/vllm_caa_gpu_e2e.py tests/test_vllm_caa_hooks.py
```

Run the optional GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
python examples/vllm_caa_gpu_e2e.py \
  --model /path/to/model \
  --tensor-parallel-size 2 \
  --layer 12 \
  --multiplier 0.0 \
  --vector-value 0.0 \
  --output /tmp/vllm_caa_gpu_e2e.json \
  --monitor-output /tmp/vllm_caa_gpu_e2e.gpu.csv
```

The smoke test records:

- baseline output
- steered output
- restored output after clearing hooks
- per-worker install/clear results
- per-worker hook call stats
- GPU monitor CSV

With `multiplier=0.0`, the hook should be a no-op and restored output should match baseline.

With a nonzero vector/multiplier, the script is a hook wiring smoke test, not a semantic-quality benchmark.

## Notes

- vLLM internals change across releases. The hook uses worker RPC through `llm.llm_engine.collective_rpc` and supports common worker model layouts.
- The hook preserves tuple tails from decoder layers.
- Clearing removes configured vectors but preserves call counters for inspection.
- Validation scripts avoid bundled datasets and large result artifacts.
