#!/usr/bin/env python
# Tests steer/vllm_caa_hooks.py for installing CAA activation additions on vLLM-style decoder layers.

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "steer" / "vllm_caa_hooks.py"
SPEC = importlib.util.spec_from_file_location("vllm_caa_hooks", MODULE_PATH)
vllm_caa_hooks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(vllm_caa_hooks)
clear_vllm_caa_hooks = vllm_caa_hooks.clear_vllm_caa_hooks
apply_caa_to_vllm_workers = vllm_caa_hooks.apply_caa_to_vllm_workers
clear_caa_from_vllm_workers = vllm_caa_hooks.clear_caa_from_vllm_workers
install_vllm_caa_hooks = vllm_caa_hooks.install_vllm_caa_hooks
install_caa_with_vllm_rpc = vllm_caa_hooks.install_caa_with_vllm_rpc
clear_caa_with_vllm_rpc = vllm_caa_hooks.clear_caa_with_vllm_rpc
get_caa_hook_stats_with_vllm_rpc = vllm_caa_hooks.get_caa_hook_stats_with_vllm_rpc


class FakeVector:
    def __init__(self, values):
        self.values = list(values)
        self.device = "cpu"
        self.dtype = "float32"

    def to(self, device=None, dtype=None):
        converted = FakeVector(self.values)
        converted.device = device or self.device
        converted.dtype = dtype or self.dtype
        return converted

    def __mul__(self, multiplier):
        return FakeVector([value * multiplier for value in self.values])

    __rmul__ = __mul__


class FakeMatrix:
    def __init__(self, rows, device="cuda:0", dtype="bfloat16"):
        self.rows = [list(row) for row in rows]
        self.device = device
        self.dtype = dtype

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return FakeMatrix(
                [[value + other for value in row] for row in self.rows],
                device=self.device,
                dtype=self.dtype,
            )
        if isinstance(other, FakeVector):
            return FakeMatrix(
                [
                    [value + other.values[col_idx] for col_idx, value in enumerate(row)]
                    for row in self.rows
                ],
                device=self.device,
                dtype=self.dtype,
            )
        raise TypeError(other)

    def tolist(self):
        return self.rows

    def __eq__(self, other):
        return isinstance(other, FakeMatrix) and self.rows == other.rows


class FakeDecoderLayer:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, positions, hidden_states, residual):
        return hidden_states + self.offset, residual


class FakeInnerModel:
    def __init__(self):
        self.layers = [FakeDecoderLayer(1.0), FakeDecoderLayer(2.0), FakeDecoderLayer(4.0)]


class FakeVllmModel:
    def __init__(self):
        self.model = FakeInnerModel()


class FakeLLM:
    def __init__(self, model):
        self.model = model
        self.applied_callable_names = []

    def apply_model(self, fn):
        self.applied_callable_names.append(getattr(fn, "__name__", fn.__class__.__name__))
        return [fn(self.model)]


class FakeWorker:
    def __init__(self, model):
        self._model = model

    def get_model(self):
        return self._model


class FakeRunnerWorker:
    def __init__(self, model):
        self.model_runner = type("Runner", (), {"model": model})()


class FakeEngine:
    def __init__(self, workers):
        self.workers = workers
        self.call_names = []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.call_names.append(getattr(method, "__name__", method.__class__.__name__))
        kwargs = kwargs or {}
        return [method(worker, *args, **kwargs) for worker in self.workers]


class FakeRpcLLM:
    def __init__(self, workers):
        self.llm_engine = FakeEngine(workers)


def fake_generate(model, hidden):
    residual = None
    for layer in model.model.layers:
        hidden, residual = layer(None, hidden, residual)
    return hidden


def test_install_vllm_caa_hooks_adds_vectors_only_to_configured_layers():
    model = FakeVllmModel()
    hidden = FakeMatrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    residual = FakeMatrix([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    vector = FakeVector([0.0, 1.0, 2.0, 3.0])

    installed = install_vllm_caa_hooks(
        model,
        layer_vectors={1: vector},
        multipliers={1: 3.0},
    )

    assert installed == [1]
    layer0_output, _ = model.model.layers[0](None, hidden, residual)
    layer1_output, layer1_residual = model.model.layers[1](None, hidden, residual)
    assert layer0_output.tolist() == [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    assert layer1_output.tolist() == [[2.0, 5.0, 8.0, 11.0], [2.0, 5.0, 8.0, 11.0]]
    assert layer1_residual is residual


def test_install_vllm_caa_hooks_preserves_tuple_tail_and_can_clear_hooks():
    class TupleTailLayer:
        def __call__(self, positions, hidden_states, residual):
            return hidden_states + 5.0, residual, "tail"

    model = FakeVllmModel()
    model.model.layers[2] = TupleTailLayer()
    hidden = FakeMatrix([[0.0, 0.0, 0.0]])
    vector = FakeVector([1.0, -1.0, 0.5])

    install_vllm_caa_hooks(model, layer_vectors={2: vector}, multipliers={2: 2.0})

    output = model.model.layers[2](None, hidden, None)
    assert output[0].tolist() == [[7.0, 3.0, 6.0]]
    assert output[1:] == (None, "tail")

    cleared = clear_vllm_caa_hooks(model)

    assert cleared == [2]
    output_after_clear = model.model.layers[2](None, hidden, None)
    assert output_after_clear[0].tolist() == [[5.0, 5.0, 5.0]]
    assert output_after_clear[1:] == (None, "tail")


def test_install_vllm_caa_hooks_rejects_missing_layers_attribute():
    try:
        install_vllm_caa_hooks(object(), layer_vectors={0: FakeVector([1.0, 1.0])})
    except ValueError as exc:
        assert "model.layers" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported vLLM model layout")


def test_fake_generation_output_is_invariant_after_install_and_clear():
    model = FakeVllmModel()
    llm = FakeLLM(model)
    hidden = FakeMatrix([[1.0, 2.0, 3.0]])
    original_output = fake_generate(model, hidden)

    apply_result = apply_caa_to_vllm_workers(
        llm,
        layer_vectors={0: FakeVector([9.0, 9.0, 9.0])},
        multipliers={0: 2.0},
    )
    steered_output = fake_generate(model, hidden)
    clear_result = clear_caa_from_vllm_workers(llm)
    restored_output = fake_generate(model, hidden)

    assert apply_result == [[0]]
    assert clear_result == [[0]]
    assert steered_output != original_output
    assert restored_output == original_output
    assert "<lambda>" not in llm.applied_callable_names


def test_worker_rpc_install_and_clear_updates_all_worker_models():
    worker_models = [FakeVllmModel(), FakeVllmModel()]
    llm = FakeRpcLLM([FakeWorker(worker_models[0]), FakeRunnerWorker(worker_models[1])])
    hidden = FakeMatrix([[0.0, 0.0, 0.0]])

    install_result = install_caa_with_vllm_rpc(
        llm,
        layer_vectors={1: FakeVector([1.0, 2.0, 3.0])},
        multipliers={1: 4.0},
    )
    worker_outputs = [fake_generate(model, hidden).tolist() for model in worker_models]
    clear_result = clear_caa_with_vllm_rpc(llm)
    restored_outputs = [fake_generate(model, hidden).tolist() for model in worker_models]

    assert install_result == [[1], [1]]
    assert clear_result == [[1], [1]]
    assert worker_outputs == [[[11.0, 15.0, 19.0]], [[11.0, 15.0, 19.0]]]
    assert restored_outputs == [[[7.0, 7.0, 7.0]], [[7.0, 7.0, 7.0]]]
    assert "<lambda>" not in llm.llm_engine.call_names


def test_worker_rpc_install_fails_on_partial_worker_update_by_default():
    llm = FakeRpcLLM([FakeWorker(FakeVllmModel()), FakeWorker(object())])

    try:
        install_caa_with_vllm_rpc(llm, layer_vectors={1: FakeVector([1.0, 2.0, 3.0])})
    except RuntimeError as exc:
        assert "worker" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when a worker cannot install CAA hooks")


def test_worker_rpc_stats_count_hook_calls_before_clear():
    model = FakeVllmModel()
    llm = FakeRpcLLM([FakeWorker(model)])
    hidden = FakeMatrix([[0.0, 0.0, 0.0]])

    install_caa_with_vllm_rpc(
        llm,
        layer_vectors={1: FakeVector([1.0, 2.0, 3.0])},
        multipliers={1: 0.0},
    )
    fake_generate(model, hidden)
    stats_before_clear = get_caa_hook_stats_with_vllm_rpc(llm)
    clear_caa_with_vllm_rpc(llm)
    stats_after_clear = get_caa_hook_stats_with_vllm_rpc(llm)

    assert stats_before_clear == [{1: {"calls": 1, "configured": True}}]
    assert stats_after_clear == [{1: {"calls": 1, "configured": False}}]


def test_zero_multiplier_hook_counts_call_without_replacing_output_object():
    class IdentityLayer:
        def __call__(self, positions, hidden_states, residual):
            return hidden_states, residual

    model = FakeVllmModel()
    model.model.layers[0] = IdentityLayer()
    hidden = FakeMatrix([[0.0, 0.0, 0.0]])
    vector = FakeVector([1.0, 2.0, 3.0])

    install_vllm_caa_hooks(model, layer_vectors={0: vector}, multipliers={0: 0.0})

    output, _ = model.model.layers[0](None, hidden, None)
    stats = vllm_caa_hooks.get_vllm_caa_hook_stats(model)

    assert output is hidden
    assert stats == {0: {"calls": 1, "configured": True}}


def test_old_single_gpu_and_vllm_hook_style_add_same_vector():
    model_a = FakeVllmModel()
    model_b = FakeVllmModel()
    hidden = FakeMatrix([[0.0, 0.0, 0.0]])
    vector = FakeVector([0.5, 1.5, -2.0])

    install_vllm_caa_hooks(model_a, layer_vectors={0: vector}, multipliers={0: 2.0})
    install_caa_with_vllm_rpc(
        FakeRpcLLM([FakeWorker(model_b)]),
        layer_vectors={0: vector * 2.0},
        multipliers={0: 1.0},
    )

    assert fake_generate(model_a, hidden) == fake_generate(model_b, hidden)
