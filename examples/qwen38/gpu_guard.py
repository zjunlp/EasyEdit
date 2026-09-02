"""Refuse to start 27B jobs when every GPU is below the free-memory floor.

Scripts import this *before* `import torch` so CUDA_VISIBLE_DEVICES is set
first. nvidia-smi is read-only: no kill, no niceness, no preemption.
"""
import subprocess

MIN_FREE_GB = 66.0


def query_gpus():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc

    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, used, free = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "free_gb": int(free) / 1024,
            }
        )
    return gpus


def pick_gpu(min_free_gb=MIN_FREE_GB):
    """Return (index, free_gb) of the emptiest card, or raise if all are busy."""
    gpus = query_gpus()
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    best = max(gpus, key=lambda g: g["free_gb"])
    if best["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"largest free GPU{best['index']} has only {best['free_gb']:.1f}GB "
            f"< {min_free_gb:.1f}GB; refusing to run so existing jobs are not disturbed"
        )
    return best["index"], best["free_gb"]
