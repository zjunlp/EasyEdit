# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for making ``torch.nn.Module`` subclass instances stateless."""

import abc as _abc
from collections import OrderedDict as _OrderedDict
from contextlib import contextmanager as _contextmanager
import typing as _typing
import weakref as _weakref
import warnings as _warnings

import torch as _torch

# from higher import utils as _utils
from .higher_utils.utils import _find_param_in_list, _get_param_mapping, get_func_params

# ==============================================================================
# Helper functions and attributes for MonkeyPatch modules.
# ==============================================================================

_internal_attrs = {
    '_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks',
    '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks',
    '_modules'
}

_BufferType = _typing.Dict[str, _typing.Optional[_torch.Tensor]]


@_contextmanager
def _modify_internally(fmodule):
    fmodule._being_modified_internally = True
    yield
    fmodule._being_modified_internally = False


def _patched_parameters(
    self, recurse: bool = True, time: _typing.Optional[int] = None
) -> _typing.Iterable[_torch.Tensor]:
    r"""Returns an iterator over monkey patched module fast parameters.

    Args:
        recurse (bool): if True, then yields fast parameters of this module
            and all submodules. Otherwise, this *still* yields parameters of
            this module and all submodules, and raises a warning. This keyword
            exists only to satisfy API compatibility with
            ``torch.nn.Module.parameters``.
        time (int or None): if None, the most recent fast parameters are
            provided. The int provided stands for the number of steps since the
            module was created. *Note* that the step counter is incremented
            every time parameters are updated, so this may not align with number
            of training or evaluations steps.

    Yields:
        Parameter: module fast weights.
    """
    if getattr(self, "_fast_params", None) is None:
        raise Exception(
            "Tried to get fast weights of a monkey patched module which does "
            "not encapsulate fast weights."
        )

    if not recurse:
        _warnings.warn(
            "Calling parameters with recurse=False on a monkey patched module "
            "still returns all the fast weights of of nested patched modules."
        )

    time = -1 if time is None else time

    if not self.track_higher_grads and time not in (-1, 0):
        raise ValueError(
            "The patched model is not tracking higher gradients. Only the "
            "latest parameters are available."
        )

    return iter(self._fast_params[time])


class _MonkeyPatchBase(_abc.ABC, _torch.nn.Module):
    @_abc.abstractmethod
    def __init__(self) -> None:
        self._param_mapping: _typing.List[int] = []
        self._being_modified_internally: bool = True
        self._track_higher_grads: bool = True

    def forward(self):
        raise NotImplementedError(
            "The monkey-patching logic has failed to override self.forward "
            "on the new module, or you tried calling forward on a patched "
            "version of a module which doesn't have forward (e.g. ModuleList)."
        )

    def _expand_params(
        self, params: _typing.List[_torch.Tensor]
    ) -> _typing.List[_torch.Tensor]:
        expanded = []
        for index in self._param_mapping:
            expanded.append(params[index])
        return expanded

    @property
    def init_fast_params(self):
        if not self.track_higher_grads:
            raise Exception(
                "Cannot get initial parameters when not tracking higher "
                "gradients."
            )
        return self._fast_params[0]

    @property
    def fast_params(self):
        return None if self._fast_params is None else self._fast_params[-1]

    @fast_params.setter
    def fast_params(self, value):
        value = list(value)
        if self._fast_params is None:
            self._fast_params = []
        if self.track_higher_grads:
            self._fast_params.append(value)
        else:
            self._fast_params[0] = value

    @property
    def track_higher_grads(self):
        return self._track_higher_grads

    @track_higher_grads.setter
    def track_higher_grads(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                "Expected boolean argument. Got: {}.".format(type(value))
            )
        self._track_higher_grads = value


def buffer_sync(
    module: _torch.nn.Module,
    fmodule: _MonkeyPatchBase,
    device: _typing.Optional[_torch.device] = None
) -> None:
    r"""One off sync (copy) of buffers in ``fmodule`` with those from ``module``.
    """
    for key, value in module._buffers.items():
        if not _torch.is_tensor(value):
            fmodule._buffers[key] = value
        elif device is None:
            fmodule._buffers[key] = value.clone().detach()
        else:
            fmodule._buffers[key] = value.clone().detach().to(device)

    for name, child in module._modules.items():
        if child == None: continue
        if name in fmodule._modules:
            buffer_sync(child, fmodule._modules[name], device)
        else:
            raise KeyError(
                "Did not find expected submodule "
                "{} of monkey-patched module {}.".format(name, fmodule)
            )


# ==============================================================================
# Helper class to use instead of actual torch.nn.Parameters when patching.
# ==============================================================================


class _ParameterPlaceholder():
    def __init__(self, name: str) -> None:
        self._param_name = name

    def __repr__(self) -> str:
        return 'Parameter placeholder ("{}")'.format(self._param_name)


_ParameterPlaceholder.__name__ = "ParameterPlaceholder"
_ParameterPlaceholder.__qualname__ = "ParameterPlaceholder"

# ==============================================================================
# Helper function for recursively patching submodules.
# ==============================================================================


def _make_functional(
    module: _torch.nn.Module,
    params_box: _typing.Sequence[_typing.Optional[_typing.List[_torch.Tensor]]],
    params_offset: int,
    root_patched: _typing.Optional[_MonkeyPatchBase] = None,
) -> _typing.Tuple[int, _MonkeyPatchBase, _typing.Type[_MonkeyPatchBase]]:

    if isinstance(module, _MonkeyPatchBase):
        raise ValueError(
            "Monkey-patching monkey-patched modules is untested uncharted "
            "territory, so we're going to assume it's done in error. If you "
            "are doing this intentionally and need this to be supported, "
            "contact the developers of this library."
        )

    param_names = list(
        name for name in module._parameters.keys()
        if module._parameters[name] is not None
    )

    _ModuleType: _typing.Type[_torch.nn.Module] = module.__class__

    # type checking of next line disabled as mypy is iffy with dynamic types
    class MonkeyPatched(_ModuleType, _MonkeyPatchBase):  # type: ignore
        _wrapped_name = type(module).__name__

        def __init__(self, original_params, root) -> None:
            _torch.nn.Module.__init__(self)
            _MonkeyPatchBase.__init__(self)
            self._root_ref = _weakref.ref(root) if root else None

            self._fast_params = None
            self._param_names = param_names

            self._original_params = original_params

            # for pretty printing
            self._parameters = _OrderedDict(
                (name, _ParameterPlaceholder(name))
                for name in self._param_names
            )
            self._modules: _typing.Dict[str, _MonkeyPatchBase] = _OrderedDict()

        @property
        def direct_submodule_call(self):
            return params_box[0] is None

        @property
        def is_root(self):
            return self._root_ref is None

        @property
        def root(self):
            if self.is_root:
                return self
            else:
                return self._root_ref()

        def __setattr__(self, name, value):
            def remove_from(*dicts):
                for d in dicts:
                    if name in d:
                        del d[name]

            params = self.__dict__.get('_parameters')
            if params is not None and name in params:
                if not isinstance(value, _torch.Tensor):
                    raise TypeError("Require Tensor as fast weights. "
                                    "Got {}".format(_torch.typename(value)))

                if not self._being_modified_internally:
                    # Additional behaviour for when fast weights are being
                    # directly modified goes here:
                    old_value = self._parameters[name]
                    fast_params = self.root.fast_params[:]
                    if not fast_params:
                        raise Exception(
                            "Cannot assign parameters to patched module which "
                            "does not have implicit fast parameters."
                        )
                    replacement_index = _find_param_in_list(
                        old_value, fast_params
                    )
                    fast_params[replacement_index] = value
                    self.update_params(fast_params)


                # Change parameters in place, usually during boxed_forward pass
                self._parameters[name] = value
            else:
                modules = self.__dict__.get('_modules')
                if isinstance(value, _torch.nn.Module):
                    if modules is None:
                        raise AttributeError(
                            "cannot assign module before Module.__init__() "
                            "call"
                        )
                    remove_from(self.__dict__, self._parameters, self._buffers)
                    modules[name] = value
                elif modules is not None and name in modules:
                    if value is not None:
                        raise TypeError(
                            (
                                "cannot assign '{}' "
                                "as child module '{}'"
                                "(torch.nn.Module or None expected)"
                            ).format(_torch.typename(value), name)
                        )
                    modules[name] = value
                else:
                    buffers = self.__dict__.get('_buffers')
                    if buffers is not None and name in buffers:
                        if value is not None and not isinstance(
                            value, _torch.Tensor
                        ):
                            raise TypeError(
                                "cannot assign '{}' as buffer '{}' "
                                "(torch.Tensor or None expected)".format(
                                    _torch.typename(value), name
                                )
                            )
                        buffers[name] = value
                    else:
                        object.__setattr__(self, name, value)

    MonkeyPatched.__name__ = "InnerFunctional" + type(module).__name__
    MonkeyPatched.__qualname__ = MonkeyPatched.__name__

    fmodule = MonkeyPatched(module.parameters(), root=root_patched)

    # If a root module hasn't been defined yet, this fmodule is the root
    if not root_patched:
        root_patched = fmodule

    # use 1 as dummy list item since we are only counting
    num_params = len([1 for p in module._parameters.values() if p is not None])

    # Copy over all attributes
    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue
        setattr(fmodule, name, attr)

    # Deal with "None"-style params
    with _modify_internally(fmodule):
        for name, attr in module.__dict__['_parameters'].items():
            if isinstance(attr, _torch.nn.Parameter):
                continue
            else:
                setattr(fmodule, name, attr)

    child_params_offset = params_offset + num_params
    for name, child in module._modules.items():
        if child == None: continue
        child_params_offset, fchild, _ = _make_functional(
            child, params_box, child_params_offset, root_patched
        )
        fmodule._modules[name] = fchild
        setattr(fmodule, name, fchild)

    true_forward = type(module).forward

    def patched_forward(self, *args, params=None, **kwargs):
        if self.direct_submodule_call:
            # If submodule was called directly, run intialisation that happens
            # at top level call. If *full set of params* is provided here, it 
            # will use those. If not, it will fall back on fast weights.
            # In the future, we should be able to support passing only the 
            # submodule (+ children) weights here, but that's not simple.
            self.root._refill_params_box(params)

        with _modify_internally(self):
            for name, param in zip(
                self._param_names,
                params_box[0][params_offset:params_offset + num_params]
            ):
                setattr(self, name, param)

            # This snippet deals with torch.nn.{RNN,GRU,LSTM}
            if hasattr(self, "_flat_weights_names"):
                self._flat_weights = [
                    self._parameters[wn] for wn in self._flat_weights_names
                ]

        # Call true_forward after some checks
        with _warnings.catch_warnings():

            # If running RNNs on GPU, surpress the warnings due to flattening
            # not happening here. Maybe we should raise a warning of our own?
            is_RNN = isinstance(module, _torch.nn.RNNBase)
            if is_RNN and _torch.cuda.is_available():
                _warnings.simplefilter("ignore", category=UserWarning)
            
            return true_forward(self, *args, **kwargs)

    setattr(MonkeyPatched, "forward", patched_forward)

    def flatten_parameters(self):
        return  # no-op

    # This (hopefully) avoids trouble on GPU with torch.nn.{RNN,GRU,LSTM}
    if hasattr(module, "flatten_parameters"):
        setattr(MonkeyPatched, "flatten_parameters", flatten_parameters)

    return child_params_offset, fmodule, type(fmodule)


def _update_patched_params(
    fmodule: _MonkeyPatchBase,
    params_box: _typing.Sequence[_typing.List[_torch.Tensor]],
    params_offset: int
) -> int:
    num_params = len([1 for p in fmodule._parameters.values() if p is not None])
    child_params_offset = params_offset + num_params
    for name, child in fmodule._modules.items():
        child_params_offset = _update_patched_params(
            child, params_box, child_params_offset
        )

    with _modify_internally(fmodule):
        for name, param in zip(
            fmodule._param_names,
            params_box[0][params_offset:params_offset + num_params]
        ):
            setattr(fmodule, name, param)
    return child_params_offset


# ==============================================================================
# The main function which does the monkey patching.
# ==============================================================================
_EncapsulatorType = _typing.Optional[
    _typing.Callable[[_MonkeyPatchBase, _torch.nn.Module], None]]


def make_functional(
    module: _torch.nn.Module,
    encapsulator: _EncapsulatorType = None
) -> _MonkeyPatchBase:
    r"""Returns a stateless version of an ``nn.Module`` instance."""
    params_box = [None]
    _, fmodule, MonkeyPatched = _make_functional(module, params_box, 0)
    top_name = "Functional" + MonkeyPatched._wrapped_name
    MonkeyPatched.__name__ = MonkeyPatched.__qualname__ = top_name

    MonkeyPatched.boxed_forward = MonkeyPatched.forward

    param_mapping = _get_param_mapping(module, [], [])
    setattr(fmodule, "_param_mapping", param_mapping)

    def _refill_params_box(self, params):
        if params is not None:
            self.fast_params = params  # update view on latest fast params
        elif self.fast_params is None:
            raise ValueError(
                "params keyword must be provided if patched module not "
                "tracking its own fast parameters"
            )

        # Copy fast parameters into params_box for use in boxed_forward
        params_box[0] = self._expand_params(self.fast_params)


    def _patched_forward(self, *args, params=None, **kwargs):
        self._refill_params_box(params)

        output = self.boxed_forward(*args, **kwargs)
        
        # Clean up
        params_box[0] = None
        
        return output

    def _update_params(self, params):
        self.fast_params = params
        params = self._expand_params(params)
        _update_patched_params(self, [params], 0)

    setattr(MonkeyPatched, "forward", _patched_forward)
    setattr(MonkeyPatched, "parameters", _patched_parameters)
    setattr(MonkeyPatched, "update_params", _update_params)
    setattr(MonkeyPatched, "_refill_params_box", _refill_params_box)

    if encapsulator is not None:
        encapsulator(fmodule, module)

    return fmodule


# ==============================================================================
# Convenience functions and decorators for hiding away a lot of the complexity
# of creating patched modules, taking their parameters, and linking patched
# modules to a differentiable optimizer.
# ==============================================================================


def monkeypatch(
    module: _torch.nn.Module,
    device: _typing.Optional[_torch.device] = None,
    copy_initial_weights: bool = True,
    track_higher_grads: bool = True,
    in_place: bool = False
) -> _MonkeyPatchBase:
    r"""Create a monkey-patched stateless version of a module.

    This function produces a monkey-patched version of a module, and returns a
    copy of its parameters for use as fast weights. Where the original module
    or any of its submodules have state (e.g. batch norm), this will be copied
    too, but further updates (e.g. during inner loop training) will cause these
    to diverge without changing the state of the original module.

    Args:
        module: a ``torch.nn.Module`` subclass instance.
        device (optional): a device to cast the fast weights and state to.
        copy_initial_weights: if True, the weights of the patched module are
            copied to form the initial weights of the patched module, and thus
            are not part of the gradient tape when unrolling the patched module.
            If this is set to False, the actual module weights will be the
            initial weights of the patched module. This is useful when doing
            MAML, for example.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows ``monkeypatch`` to be used in "test mode", without
            potentially tracking higher order gradients. This can be useful when
            running the training loop at test time, e.g. in k-shot learning
            experiments, without incurring a significant memory overhead.

    Returns:
        ``fmodule``: a "stateless" version of the original module, for which calls
        to forward take the additional kwarg-only parameter ``params``, which
        should be a list of torch tensors requiring gradients, ideally
        provided by this function (see below) or by an update step from one
        of the optimizers in ``higher.optim``.
    """

    def encapsulator(
        fmodule: _MonkeyPatchBase, module: _torch.nn.Module
    ) -> None:
        if copy_initial_weights and not in_place:
            params = get_func_params(module, device=device)
        else:
            if in_place:
                params = [
                    p if device is None else p.to(device)
                    for p in module.parameters()
                ]
            else:  # Standard behavior
                params = [
                    p.clone() if device is None else p.clone().to(device)
                    for p in module.parameters()
                ]
        buffer_sync(module, fmodule, device)
        fmodule.update_params(params)

    fmodule = make_functional(module, encapsulator=encapsulator)
    fmodule.track_higher_grads = track_higher_grads

    return fmodule
