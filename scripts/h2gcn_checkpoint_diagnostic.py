"""Checkpoint-copy treatments for the bounded H2GCN CPU performance diagnostic.

This helper does not train, select a checkpoint, change an optimizer, or alter
the original model implementation. Copy timings exclude static graph fingerprint validation.
Call ``check_static`` at each trial boundary, before using a new model instance,
and after the final capture. ``restore`` performs those checks itself.
"""
from __future__ import annotations

import hashlib
import json
import time
from types import MappingProxyType
import weakref

import torch

from experiments.prospective_models import H2GCNModel


MODES = ("full_state_copy", "static_adjacency_once")
PARAMETER_NAMES = ("embedding.weight", "embedding.bias", "classifier.weight", "classifier.bias")
BUFFER_NAMES = ("one_hop", "two_hop")
STATE_NAMES = frozenset(PARAMETER_NAMES + BUFFER_NAMES)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _metadata(value):
    return {"shape": tuple(value.shape), "dtype": str(value.dtype), "layout": str(value.layout)}


def tensor_fingerprint(value):
    """Hash tensor values and metadata without allocating a dense graph."""
    value = value.detach().cpu()
    if value.layout == torch.sparse_coo:
        value = value.coalesce()
        return _json_hash({**_metadata(value), "indices": tensor_fingerprint(value.indices()),
                           "values": tensor_fingerprint(value.values())})
    if value.layout != torch.strided:
        raise ValueError("only dense parameters and COO adjacency buffers are supported")
    value = value.contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(_metadata(value), sort_keys=True).encode())
    # A byte view avoids a graph-sized temporary ``numpy().tobytes()`` copy.
    if value.numel():
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def state_fingerprint(state):
    return _json_hash({name: tensor_fingerprint(value) for name, value in sorted(state.items())})


def _buffer_token(value):
    """Constant-size mutation guard; full content validation is separate."""
    return (id(value), value._version, value._indices().data_ptr(), value._indices()._version,
            value._values().data_ptr(), value._values()._version)


def _storages(value):
    if value.layout == torch.sparse_coo:
        yield from _storages(value._indices())
        yield from _storages(value._values())
    elif value.layout == torch.strided:
        storage = value.untyped_storage()
        yield (str(value.device), storage.data_ptr(), storage.nbytes()), storage.nbytes()
    else:
        raise ValueError("unsupported tensor layout")


def _unique_bytes(values):
    storages = {}
    for value in values:
        storages.update(_storages(value))
    return sum(storages.values())


class CheckpointStore:
    """Four immutable CPU checkpoint slots under one fixed copy treatment.

    ``snapshot`` exposes a read-only mapping for caller comparisons. Its tensor
    values must also be treated as read-only; mutating a view is not supported.
    Bank corruption is checked at validation/restore boundaries. Storage metrics
    describe tensors retained by this store, excluding model, optimizer, Python
    overhead, and any extra references retained by callers.
    """

    def __init__(self, model, mode):
        if mode not in MODES:
            raise ValueError(f"unknown checkpoint treatment: {mode}")
        self.mode = mode
        self._slots = {}
        self._schema = None
        self._parameter_requires_grad = None
        self._validated_models = weakref.WeakKeyDictionary()
        self._static_bank = None
        self._bank_tokens = None
        self._last_capture_peak_unique_tensor_bytes = 0
        self.initial_static_copy_seconds = 0.0
        self.initial_static_copy_bytes = 0
        started = time.perf_counter()
        self._validate_model(model)
        self.initial_schema_validation_seconds = time.perf_counter() - started
        source = model.state_dict()
        started = time.perf_counter()
        self._static_fingerprint = state_fingerprint({name: source[name] for name in BUFFER_NAMES})
        self.initial_static_hash_seconds = time.perf_counter() - started
        if mode == "static_adjacency_once":
            started = time.perf_counter()
            bank = {name: source[name].detach().cpu().clone() for name in BUFFER_NAMES}
            self.initial_static_copy_seconds = time.perf_counter() - started
            self._static_bank = bank
            self.initial_static_copy_bytes = _unique_bytes(bank.values())
            self._bank_tokens = {name: _buffer_token(value) for name, value in bank.items()}
        started = time.perf_counter()
        self.check_static(model)
        self.initial_boundary_validation_seconds = time.perf_counter() - started

    @property
    def static_fingerprint(self):
        return self._static_fingerprint

    def _validate_model(self, model):
        if type(model) is not H2GCNModel:
            raise ValueError("checkpoint treatment requires the unchanged original H2GCNModel")
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        if set(parameters) != set(PARAMETER_NAMES) or set(buffers) != set(BUFFER_NAMES):
            raise ValueError("H2GCN parameter or buffer names changed")
        state = model.state_dict()
        if set(state) != STATE_NAMES:
            raise ValueError("H2GCN state keys changed")
        if model.propagation_layers != 2:
            raise ValueError("H2GCN propagation depth changed")
        for name, value in state.items():
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise ValueError(f"CPU float32 state is required: {name}")
            if name in BUFFER_NAMES:
                if (value.layout != torch.sparse_coo or not value.is_coalesced()
                        or value.dim() != 2 or value.shape[0] != value.shape[1]):
                    raise ValueError(f"coalesced square COO buffer is required: {name}")
            elif value.layout != torch.strided:
                raise ValueError(f"dense parameter is required: {name}")
        if state["one_hop"].shape != state["two_hop"].shape:
            raise ValueError("H2GCN adjacency shapes differ")
        if state["embedding.weight"].dim() != 2 or state["classifier.weight"].dim() != 2:
            raise ValueError("H2GCN parameter shapes are inconsistent")
        hidden = state["embedding.weight"].shape[0]
        if (state["embedding.bias"].shape != (hidden,)
                or state["classifier.weight"].shape[1] != 7 * hidden
                or state["classifier.bias"].shape != (state["classifier.weight"].shape[0],)):
            raise ValueError("H2GCN parameter shapes are inconsistent")
        metadata = {name: _metadata(value) for name, value in state.items()}
        requires_grad = {name: value.requires_grad for name, value in parameters.items()}
        if self._schema is None:
            self._schema = metadata
            self._parameter_requires_grad = requires_grad
        elif metadata != self._schema or requires_grad != self._parameter_requires_grad:
            raise ValueError("H2GCN state shapes, dtypes, layouts or trainable flags changed")

    def _validate_snapshot(self, state):
        if set(state) != STATE_NAMES:
            raise ValueError("checkpoint state keys changed")
        if any(value.device.type != "cpu" or _metadata(value) != self._schema[name]
               for name, value in state.items()):
            raise ValueError("checkpoint tensor shape, dtype, layout or device changed")
        if any(not state[name].is_coalesced() for name in BUFFER_NAMES):
            raise ValueError("checkpoint adjacency coalescing changed")

    def _validate_bank(self):
        if self._static_bank is not None:
            if set(self._static_bank) != set(BUFFER_NAMES):
                raise ValueError("private static adjacency bank keys changed")
            if any(_metadata(value) != self._schema[name] or value.device.type != "cpu"
                   for name, value in self._static_bank.items()):
                raise ValueError("private static adjacency bank metadata changed")
            if any(not value.is_coalesced() for value in self._static_bank.values()):
                raise ValueError("private static adjacency bank coalescing changed")
            if state_fingerprint(self._static_bank) != self._static_fingerprint:
                raise ValueError("private static adjacency bank is corrupt")

    def _validate_stored_static(self, state):
        if self._static_bank is not None:
            # The bank was already checked once. Verify sharing without hashing
            # the same graph again for each parameter checkpoint.
            if any(state[name] is not self._static_bank[name] for name in BUFFER_NAMES):
                raise ValueError("checkpoint no longer references the private static bank")
        elif state_fingerprint({name: state[name] for name in BUFFER_NAMES}) != self._static_fingerprint:
            raise ValueError("stored checkpoint adjacency buffers changed")

    def check_static(self, model):
        """Fully validate static values outside copy timing; register this model."""
        self._validate_model(model)
        buffers = dict(model.named_buffers())
        if state_fingerprint(buffers) != self._static_fingerprint:
            raise ValueError("source adjacency buffers changed")
        self._validate_bank()
        for state in self._slots.values():
            self._validate_snapshot(state)
            self._validate_stored_static(state)
        self._validated_models[model] = {name: _buffer_token(value) for name, value in buffers.items()}
        if self._static_bank is not None:
            self._bank_tokens = {name: _buffer_token(value) for name, value in self._static_bank.items()}
        return self._static_fingerprint

    def _check_capture_tokens(self, model):
        if model not in self._validated_models:
            raise ValueError("call check_static(model) before capturing a new model instance")
        current = {name: _buffer_token(value) for name, value in model.named_buffers()}
        if current != self._validated_models[model]:
            raise ValueError("source buffer identity/version changed; full static validation is required")
        if self._static_bank is not None:
            if (set(self._static_bank) != set(BUFFER_NAMES)
                    or any(_metadata(value) != self._schema[name] for name, value in self._static_bank.items())):
                raise ValueError("private static bank metadata changed")
            current_bank = {name: _buffer_token(value) for name, value in self._static_bank.items()}
            if current_bank != self._bank_tokens:
                raise ValueError("private static bank identity/version changed; full static validation is required")

    @staticmethod
    def _validate_slot(slot):
        if type(slot) is not int or slot not in range(4):
            raise ValueError("checkpoint slot must be an integer in 0..3")

    def capture(self, model, slot):
        """Copy a state without modifying model/RNG and atomically replace a slot."""
        self._validate_slot(slot)
        started = time.perf_counter()
        self._validate_model(model)
        self._check_capture_tokens(model)
        copied = {}
        parameter_seconds = buffer_seconds = 0.0
        # state_dict order preserves the baseline's buffer-before-parameter copy
        # order. The old slot remains resident until every new copy is complete.
        for name, value in model.state_dict().items():
            if name in BUFFER_NAMES and self._static_bank is not None:
                copied[name] = self._static_bank[name]
                continue
            copy_started = time.perf_counter()
            clone = value.detach().cpu().clone()
            elapsed = time.perf_counter() - copy_started
            copied[name] = clone
            if name in PARAMETER_NAMES:
                parameter_seconds += elapsed
            else:
                buffer_seconds += elapsed
        peak = self._storage_metrics(extra_state=copied)["total_unique_tensor_bytes"]
        replacement = slot in self._slots
        self._slots[slot] = copied
        self._last_capture_peak_unique_tensor_bytes = peak
        metrics = self.storage_metrics()
        return {"mode": self.mode, "slot": slot, "replacement": replacement,
                "parameter_copy_seconds": parameter_seconds,
                "static_buffer_copy_seconds": buffer_seconds,
                "capture_seconds": time.perf_counter() - started,
                "parameter_copy_bytes": _unique_bytes(copied[name] for name in PARAMETER_NAMES),
                "static_buffer_copy_bytes": (_unique_bytes(copied[name] for name in BUFFER_NAMES)
                                              if self.mode == "full_state_copy" else 0),
                "retained_checkpoint_count": len(self._slots),
                "peak_retained_tensor_bytes": peak,
                "post_capture_retained_tensor_bytes": metrics["total_unique_tensor_bytes"]}

    def snapshot(self, slot):
        """Return a read-only full-state mapping view; do not mutate its tensors."""
        self._validate_slot(slot)
        if slot not in self._slots:
            raise KeyError(f"checkpoint slot {slot} is empty")
        self._validate_bank()
        state = self._slots[slot]
        self._validate_snapshot(state)
        self._validate_stored_static(state)
        return MappingProxyType(dict(state))

    def restore(self, model, slot):
        """Strictly restore all parameters and buffers after static validation."""
        self.check_static(model)
        state = self.snapshot(slot)
        model.load_state_dict(state, strict=True)
        self.check_static(model)

    def _storage_metrics(self, extra_state=None):
        states = list(self._slots.values())
        if extra_state is not None:
            states.append(extra_state)
        parameters = [state[name] for state in states for name in PARAMETER_NAMES]
        buffers = [state[name] for state in states for name in BUFFER_NAMES]
        if self._static_bank is not None:
            buffers.extend(self._static_bank.values())
        parameter_bytes, buffer_bytes = _unique_bytes(parameters), _unique_bytes(buffers)
        return {"mode": self.mode, "slots": sorted(self._slots),
                "retained_checkpoint_count": len(self._slots),
                "retained_parameter_bytes": parameter_bytes,
                "retained_static_buffer_bytes": buffer_bytes,
                "total_unique_tensor_bytes": _unique_bytes(parameters + buffers),
                "static_buffer_replicas": (1 if self._static_bank is not None else len(states)),
                "last_capture_peak_unique_tensor_bytes": self._last_capture_peak_unique_tensor_bytes,
                "scope": "retained tensor storages only; excludes model, optimizer, Python overhead and caller-held views"}

    def storage_metrics(self):
        """Measure unique retained tensor storage, not process RSS or savings."""
        return self._storage_metrics()
