#!/usr/bin/env python
# Utilities for installing CAA activation-add hooks on vLLM worker-local decoder layers.

_HOOK_ATTR = "_easyedit_caa_hook"
_ORIGINAL_ATTR = "_easyedit_caa_original"
_WRAPPER_ATTR = "_easyedit_caa_wrapper"
_CALLS_KEY = "calls"


class _CallableLayerWrapper:
    def __init__(self, layer):
        self._easyedit_wrapped_layer = layer
        setattr(self, _WRAPPER_ATTR, True)
        setattr(self, _HOOK_ATTR, {})

    def __getattr__(self, name):
        return getattr(self._easyedit_wrapped_layer, name)

    def __call__(self, *args, **kwargs):
        output = self._easyedit_wrapped_layer(*args, **kwargs)
        return _add_configured_vectors(output, getattr(self, _HOOK_ATTR))


class InstallVllmCAAHooks:
    def __init__(self, layer_vectors, multipliers=None):
        self.layer_vectors = layer_vectors
        self.multipliers = multipliers

    def __call__(self, model):
        return install_vllm_caa_hooks(model, self.layer_vectors, self.multipliers)


class ClearVllmCAAHooks:
    def __init__(self, layers=None):
        self.layers = layers

    def __call__(self, model):
        return clear_vllm_caa_hooks(model, self.layers)


class InstallVllmCAAHooksOnWorker:
    def __init__(self, layer_vectors, multipliers=None):
        self.layer_vectors = layer_vectors
        self.multipliers = multipliers

    def __call__(self, worker):
        try:
            return install_vllm_caa_hooks(_worker_model(worker), self.layer_vectors, self.multipliers)
        except Exception:
            return False


class ClearVllmCAAHooksOnWorker:
    def __init__(self, layers=None):
        self.layers = layers

    def __call__(self, worker):
        try:
            return clear_vllm_caa_hooks(_worker_model(worker), self.layers)
        except Exception:
            return False


class GetVllmCAAHookStatsOnWorker:
    def __call__(self, worker):
        try:
            return get_vllm_caa_hook_stats(_worker_model(worker))
        except Exception:
            return False


def install_vllm_caa_hooks(model, layer_vectors, multipliers=None):
    """Install CAA vector addition on a vLLM-loaded model and return touched layers."""
    layers = _find_layers(model)
    multipliers = multipliers or {}
    installed = []

    for layer_idx, vector in layer_vectors.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer index {layer_idx} is outside model.layers")

        layer = layers[layer_idx]
        layer = _ensure_layer_hook(layers, layer_idx, layer)
        hook_state = getattr(layer, _HOOK_ATTR)
        hook_state["vector"] = vector
        hook_state["multiplier"] = multipliers.get(layer_idx, 1.0)
        hook_state[_CALLS_KEY] = 0
        installed.append(layer_idx)

    return installed


def clear_vllm_caa_hooks(model, layers=None):
    """Clear EasyEdit CAA vectors from a vLLM-loaded model and return cleared layers."""
    model_layers = _find_layers(model)
    target_layers = range(len(model_layers)) if layers is None else layers
    cleared = []

    for layer_idx in target_layers:
        layer = model_layers[layer_idx]
        if hasattr(layer, _HOOK_ATTR):
            hook_state = getattr(layer, _HOOK_ATTR)
            calls = hook_state.get(_CALLS_KEY, 0)
            hook_state.clear()
            hook_state[_CALLS_KEY] = calls
            cleared.append(layer_idx)

    return cleared


def get_vllm_caa_hook_stats(model):
    """Return call counters for layers touched by EasyEdit CAA hooks."""
    model_layers = _find_layers(model)
    stats = {}
    for layer_idx, layer in enumerate(model_layers):
        if hasattr(layer, _HOOK_ATTR):
            hook_state = getattr(layer, _HOOK_ATTR)
            stats[layer_idx] = {
                "calls": hook_state.get(_CALLS_KEY, 0),
                "configured": "vector" in hook_state,
            }
    return stats


def apply_caa_to_vllm_workers(llm, layer_vectors, multipliers=None):
    """Broadcast CAA hook installation to every vLLM worker through LLM.apply_model."""
    if not hasattr(llm, "apply_model"):
        raise ValueError("vLLM LLM object does not expose apply_model")
    return llm.apply_model(InstallVllmCAAHooks(layer_vectors, multipliers))


def clear_caa_from_vllm_workers(llm, layers=None):
    """Broadcast CAA hook clearing to every vLLM worker through LLM.apply_model."""
    if not hasattr(llm, "apply_model"):
        raise ValueError("vLLM LLM object does not expose apply_model")
    return llm.apply_model(ClearVllmCAAHooks(layers))


def get_caa_hook_stats_with_vllm_rpc(llm, require_all=True):
    """Read CAA hook counters through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(GetVllmCAAHookStatsOnWorker())
    if require_all:
        _raise_on_worker_failures(results, "stats")
    return results


def install_caa_with_vllm_rpc(llm, layer_vectors, multipliers=None, require_all=True):
    """Install CAA hooks through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(
        InstallVllmCAAHooksOnWorker(layer_vectors, multipliers)
    )
    if require_all:
        _raise_on_worker_failures(results, "install")
    return results


def clear_caa_with_vllm_rpc(llm, layers=None, require_all=True):
    """Clear CAA hook state through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(ClearVllmCAAHooksOnWorker(layers))
    if require_all:
        _raise_on_worker_failures(results, "clear")
    return results


def _find_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unsupported vLLM model layout: expected model.model.layers or model.layers")


def _collective_rpc_engine(llm):
    engine = getattr(llm, "llm_engine", llm)
    if not hasattr(engine, "collective_rpc"):
        raise ValueError("vLLM object does not expose llm_engine.collective_rpc")
    return engine


def _worker_model(worker):
    if hasattr(worker, "get_model") and callable(worker.get_model):
        return worker.get_model()
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None and hasattr(model_runner, "model"):
        return model_runner.model
    if hasattr(worker, "model"):
        return worker.model
    raise ValueError("Unsupported vLLM worker layout: expected get_model(), model_runner.model, or model")


def _raise_on_worker_failures(results, action):
    failures = [
        index
        for index, result in enumerate(results)
        if isinstance(result, BaseException) or result is False or result is None
    ]
    if failures:
        raise RuntimeError(f"CAA hook {action} failed on worker indices {failures}: {results}")


def _ensure_layer_hook(layers, layer_idx, layer):
    if hasattr(layer, _HOOK_ATTR):
        return layer

    if hasattr(layer, "forward") and callable(layer.forward):
        original_forward = layer.forward

        def hooked_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            return _add_configured_vectors(output, getattr(layer, _HOOK_ATTR))

        setattr(layer, _ORIGINAL_ATTR, original_forward)
        setattr(layer, _HOOK_ATTR, {})
        layer.forward = hooked_forward
        return layer

    wrapped_layer = _CallableLayerWrapper(layer)
    layers[layer_idx] = wrapped_layer
    return wrapped_layer


def _add_configured_vectors(output, hook_state):
    if "vector" not in hook_state:
        return output

    hidden_states, tail = _split_layer_output(output)
    hook_state[_CALLS_KEY] = hook_state.get(_CALLS_KEY, 0) + 1
    multiplier = hook_state["multiplier"]
    if multiplier == 0:
        return output

    vector = _vector_like_hidden(hook_state["vector"], hidden_states)
    hidden_states = hidden_states + (vector * multiplier)
    if tail is None:
        return hidden_states
    return (hidden_states,) + tail


def _split_layer_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _vector_like_hidden(vector, hidden_states):
    if hasattr(vector, "to"):
        to_kwargs = {}
        if hasattr(hidden_states, "device"):
            to_kwargs["device"] = hidden_states.device
        if hasattr(hidden_states, "dtype"):
            to_kwargs["dtype"] = hidden_states.dtype
        return vector.to(**to_kwargs)
    return vector
