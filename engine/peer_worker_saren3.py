"""Isolated vLLM worker extension: RTX3090 trunk, GB10 resident routed experts.

This module is loaded only by the benchmark's explicit worker_extension_cls.
No installed engine files, native code, routing, or checkpoint files are modified.
"""
import os
import json
import time
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from benchmark_worker import BenchmarkWorkerExtension

_ARENAS = {}
_COUNTS = {"remote_calls": 0, "activation_bytes": 0,
           "helper_graph_captures": 0, "helper_graph_replays": 0,
           "async_event_bridges": 0, "shared_overlap_defers": 0}
_PEER_DIAGNOSTICS = os.environ.get("QWEN38_PEER_DIAGNOSTICS") == "1"
_ROUND_RECORD_FUNCTIONS = os.environ.get("QWEN38_ROUND_RECORD_FUNCTION") == "1"
_PLACEMENT = []
_SHARED_STREAMS = {}
_ROUTE_CALIBRATION = False
_ROUTES = {}
_SHARED_OVERLAP = os.environ.get("QWEN38_SHARED_OVERLAP") == "1"
_NUMERIC_ADAPTERS = {}
_HOT_CACHES = {}
_QAD_NATIVE = os.environ.get("QWEN38_QAD_NATIVE") == "1"
_QAD_HOT_CACHE_ENABLED = True
_HOT_CACHE_OWNER_BUDGET = int(float(os.environ.get("QWEN38_HOT_OWNER_BUDGET_GIB", "4")) * 1024**3)


def _shared_output_slot(module, index, output):
    outputs = getattr(module, "_qwen38_shared_outputs", None)
    if outputs is None:
        outputs = module._qwen38_shared_outputs = {}
    key = (index, tuple(output.shape), output.dtype, output.device)
    stable = outputs.get(key)
    if stable is None:
        stable = outputs[key] = torch.empty_like(output)
    return stable


def _shared_overlap_eligible(module, hidden_states, owner):
    if (not _SHARED_OVERLAP
            or getattr(module, "_qwen38_remote_routed", None) is None
            or getattr(module, "enable_dbo", True) is not False
            or hidden_states.ndim == 0
            or not 4 <= hidden_states.shape[0] <= 8
            or hidden_states.device != torch.device("cuda", owner)):
        return False
    stream = getattr(module, "_stream", None)
    if stream is None or stream.device != torch.device("cuda", owner):
        return False
    determine_order = getattr(module, "_determine_shared_experts_order", None)
    if not callable(determine_order):
        return False
    return getattr(determine_order(hidden_states), "name", None) == "MULTI_STREAM_OVERLAPPED"


def _move_trunk_state(module, owner):
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

    device = torch.device("cuda", owner)
    module._apply(lambda tensor: tensor.to(device), recurse=False)
    if isinstance(module, SharedExperts) and module._stream is not None and module._stream.device != device:
        stream = _SHARED_STREAMS.get(owner)
        if stream is None:
            stream = torch.cuda.Stream(device=owner)
            _SHARED_STREAMS[owner] = stream
        module._stream = stream
    method = getattr(module, "quant_method", None)
    if isinstance(getattr(method, "fp8_linear", None), MarlinFP8ScaledMMLinearKernel):
        module.workspace = marlin_make_workspace_new(device)
    kernel = getattr(method, "kernel", None)
    if isinstance(kernel, MarlinLinearKernel):
        kernel.workspace = marlin_make_workspace_new(device)
    for key, value in tuple(vars(module).items()):
        if isinstance(value, torch.Tensor) and value.is_cuda and value.device != device:
            setattr(module, key, value.to(device))


def _record_routes(layer, ids, owner_wait):
    name = layer._qwen38_route_name
    record = _ROUTES.get(name)
    if record is None:
        record = _ROUTES[name] = {
            "counts": [0] * layer.global_num_experts, "batches": [],
            "owner_to_host_wait_s": 0.0, "helper_step_s": 0.0, "invalid_ids": 0}
    values = ids.reshape(-1).tolist()
    for expert in values:
        if 0 <= expert < len(record["counts"]):
            record["counts"][expert] += 1
        else:
            record["invalid_ids"] += 1
    record["batches"].append({"rows": ids.shape[0], "ids": values})
    record["owner_to_host_wait_s"] += owner_wait
    return record


class TransferArena:
    def __init__(self, x, weights, ids, owner, helper):
        self.owner = owner
        self.helper = helper
        self.host_x = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        self.host_weights = torch.empty(weights.shape, dtype=weights.dtype, pin_memory=True)
        self.host_ids = torch.empty(ids.shape, dtype=ids.dtype, pin_memory=True)
        self.host_y = torch.empty_like(self.host_x, pin_memory=True)
        self.remote_x = torch.empty_like(x, device=helper)
        self.remote_weights = torch.empty_like(weights, device=helper)
        self.remote_ids = torch.empty_like(ids, device=helper)
        self.returned = torch.empty_like(x, device=owner)
        graph_mode = os.environ.get("QWEN38_HELPER_GRAPHS")
        self.graphs_enabled = graph_mode in ("1", "io") and x.shape[0] <= 16
        self.io_graphs = self.graphs_enabled and graph_mode == "io"
        self.graphs = {}
        self.capture_stream = torch.cuda.Stream(device=helper) if self.graphs_enabled else None
        self.async_transfers = os.environ.get("QWEN38_ASYNC_TRANSFERS") == "1"
        self.owner_ready = torch.cuda.Event() if self.async_transfers else None
        self.helper_ready = torch.cuda.Event() if self.async_transfers else None
        self._diagnostics = None
        self._diagnostic_capture_setup = False
        if _PEER_DIAGNOSTICS:
            input_bytes = sum(t.numel() * t.element_size() for t in (x, weights, ids))
            output_bytes = self.returned.numel() * self.returned.element_size()
            self._diagnostics = {
                "scope": {
                    "input_shape": list(x.shape),
                    "activation_dtype": str(x.dtype),
                    "route_id_shape": list(ids.shape),
                    "route_id_dtype": str(ids.dtype),
                    "route_weight_shape": list(weights.shape),
                    "route_weight_dtype": str(weights.dtype),
                    "output_shape": list(self.returned.shape),
                    "output_dtype": str(self.returned.dtype),
                    "activation_device": str(x.device),
                    "route_id_device": str(ids.device),
                    "route_weight_device": str(weights.device),
                    "host_stage_device": str(self.host_x.device),
                    "owner_device": str(self.returned.device),
                    "helper_device": str(self.remote_x.device),
                },
                "copy_ops": {
                    "owner_gpu_to_pinned_host_input": 0,
                    "pinned_host_to_helper_gpu_input": 0,
                    "helper_gpu_to_pinned_host_output": 0,
                    "pinned_host_to_owner_gpu_output": 0,
                },
                "copy_payload_bytes": {
                    "owner_gpu_to_pinned_host_input": 0,
                    "pinned_host_to_helper_gpu_input": 0,
                    "helper_gpu_to_pinned_host_output": 0,
                    "pinned_host_to_owner_gpu_output": 0,
                },
                "capture_setup_copy_ops": {
                    "pinned_host_to_helper_gpu_input": 0,
                    "helper_gpu_to_pinned_host_output": 0,
                },
                "capture_setup_copy_payload_bytes": {
                    "pinned_host_to_helper_gpu_input": 0,
                    "helper_gpu_to_pinned_host_output": 0,
                },
                "event_ops": {
                    "owner_ready_records": 0,
                    "helper_wait_owner_event_calls": 0,
                    "helper_ready_records": 0,
                    "owner_wait_helper_event_calls": 0,
                    "owner_stream_synchronize_calls": 0,
                    "helper_stream_synchronize_calls": 0,
                    "owner_wait_shared_stream_calls": 0,
                },
                "helper_graphs": {
                    "mode": graph_mode or "0",
                    "enabled": self.graphs_enabled,
                    "io_graphs": self.io_graphs,
                    "captures": 0,
                    "replays": 0,
                    "copy_ops_embedded_per_replay": 4 if self.io_graphs else 0,
                },
                "shared_overlap": {"deferred_calls": 0},
            }
            self._diagnostic_copy_bytes = {
                "owner_gpu_to_pinned_host_input": input_bytes,
                "pinned_host_to_helper_gpu_input": input_bytes,
                "helper_gpu_to_pinned_host_output": output_bytes,
                "pinned_host_to_owner_gpu_output": output_bytes,
            }

    def _count_diagnostic(self, group, key, amount=1):
        if self._diagnostics is not None:
            self._diagnostics[group][key] += amount

    def _record_diagnostic_copy(self, name, calls=1, capture_setup=False):
        if self._diagnostics is None:
            return
        ops = "capture_setup_copy_ops" if capture_setup else "copy_ops"
        payload = "capture_setup_copy_payload_bytes" if capture_setup else "copy_payload_bytes"
        self._diagnostics[ops][name] += calls
        self._diagnostics[payload][name] += self._diagnostic_copy_bytes[name]

    def _diagnostic_snapshot(self):
        if self._diagnostics is None:
            return None
        return {
            "scope": {
                key: value[:] if isinstance(value, list) else value
                for key, value in self._diagnostics["scope"].items()
            },
            "copy_ops": dict(self._diagnostics["copy_ops"]),
            "capture_setup_copy_ops": dict(self._diagnostics["capture_setup_copy_ops"]),
            "capture_setup_copy_payload_bytes": dict(
                self._diagnostics["capture_setup_copy_payload_bytes"]),
            "counter_scope": (
                "cumulative per arena; grouped input copy_ops count 3 submissions while payload_bytes sum their payload once; "
                "capture_setup counts executed warmup/restore copies, not graph templates; no duration inference"),
            "copy_payload_bytes": dict(self._diagnostics["copy_payload_bytes"]),
            "event_ops": dict(self._diagnostics["event_ops"]),
            "helper_graphs": dict(self._diagnostics["helper_graphs"]),
            "shared_overlap": dict(self._diagnostics["shared_overlap"]),
        }

    def qad_active_inputs(self, x, weights, ids, is_padding):
        if is_padding is None:
            return x, weights, ids
        if (is_padding.dtype != torch.bool or is_padding.device != ids.device
                or tuple(is_padding.shape) != (x.shape[0],)):
            raise ValueError("QAD padding mask must match the actual router rows and device")
        padding = is_padding[:, None]
        torch._assert_async(
            torch.all(torch.where(padding, ids == -1, (ids >= 0) & (ids < 512))),
            "QAD active IDs must be 0..511 and explicitly padded IDs must be -1",
        )
        buffers = getattr(self, "_qad_active_input_buffers", None)
        if buffers is None:
            buffers = self._qad_active_input_buffers = (
                torch.empty_like(x), torch.empty_like(weights), torch.empty_like(ids),
            )
            self._qad_active_input_zeros = (
                torch.zeros((), device=x.device, dtype=x.dtype),
                torch.zeros((), device=weights.device, dtype=weights.dtype),
                torch.zeros((), device=ids.device, dtype=ids.dtype),
            )
        safe_x, safe_weights, safe_ids = buffers
        # Only producer-marked inactive rows are replaced. Their original
        # scores need not be zero, so neutralize both input and contribution.
        zero_x, zero_weights, zero_ids = self._qad_active_input_zeros
        torch.where(padding, zero_x, x, out=safe_x)
        torch.where(padding, zero_weights, weights, out=safe_weights)
        torch.where(padding, zero_ids, ids, out=safe_ids)
        return safe_x, safe_weights, safe_ids

    def _apply_helper(self, layer, original):
        if self.io_graphs:
            self.remote_x.copy_(self.host_x, non_blocking=True)
            self.remote_weights.copy_(self.host_weights, non_blocking=True)
            self.remote_ids.copy_(self.host_ids, non_blocking=True)
            if self._diagnostics is not None and self._diagnostic_capture_setup:
                self._record_diagnostic_copy(
                    "pinned_host_to_helper_gpu_input", calls=3, capture_setup=True)
        value = original(layer, self.remote_x, self.remote_weights, self.remote_ids,
                         shared_experts=None, shared_experts_input=None)
        if self.io_graphs:
            self.host_y.copy_(value, non_blocking=True)
            if self._diagnostics is not None and self._diagnostic_capture_setup:
                self._record_diagnostic_copy(
                    "helper_gpu_to_pinned_host_output", capture_setup=True)
        return value

    def _expert_value(self, layer, original):
        if not self.graphs_enabled:
            return self._apply_helper(layer, original)
        cached = self.graphs.get(layer)
        if cached is None:
            current = torch.cuda.current_stream(self.helper)
            self.capture_stream.wait_stream(current)
            if self._diagnostics is not None:
                self._diagnostic_capture_setup = True
            try:
                with torch.cuda.stream(self.capture_stream):
                    for _ in range(3):
                        self._apply_helper(layer, original)
            finally:
                if self._diagnostics is not None:
                    self._diagnostic_capture_setup = False
            current.wait_stream(self.capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.capture_stream):
                value = self._apply_helper(layer, original)
            current.wait_stream(self.capture_stream)
            if not self.io_graphs:
                # Restore inputs if compute-only warmup used in-place operations.
                self.remote_x.copy_(self.host_x, non_blocking=True)
                self.remote_weights.copy_(self.host_weights, non_blocking=True)
                self.remote_ids.copy_(self.host_ids, non_blocking=True)
                if self._diagnostics is not None:
                    self._record_diagnostic_copy(
                        "pinned_host_to_helper_gpu_input", calls=3, capture_setup=True)
            from vllm.model_executor.layers.fused_moe import modular_kernel
            workspace_refs = tuple(modular_kernel.current_workspace_manager()._current_workspaces)
            cached = self.graphs[layer] = (graph, value, workspace_refs)
            _COUNTS["helper_graph_captures"] += 1
            if self._diagnostics is not None:
                self._count_diagnostic("helper_graphs", "captures")
            if self.io_graphs and _COUNTS["helper_graph_captures"] == 1:
                print({"peer_helper_io_graph_capture": True,
                       "rows": self.host_x.shape[0]}, flush=True)
        graph, value, workspace_refs = cached
        graph.replay()
        _COUNTS["helper_graph_replays"] += 1
        if self._diagnostics is not None:
            self._count_diagnostic("helper_graphs", "replays")
            if self.io_graphs:
                self._record_diagnostic_copy("pinned_host_to_helper_gpu_input", calls=3)
                self._record_diagnostic_copy("helper_gpu_to_pinned_host_output")
        return value

    def _run_deferred_shared(self, layer):
        shared = getattr(layer, "_qwen38_shared_experts", None)
        pending = getattr(shared, "_qwen38_shared_deferred", None)
        if not isinstance(pending, dict) or pending.get("input") is None:
            return None
        index = pending["index"]
        hidden_states = pending["input"]
        if (pending.get("output") is not None
                or index != shared._output_idx
                or shared._output[index] is not None
                or not _shared_overlap_eligible(shared, hidden_states, self.owner)):
            return None
        stream = shared._stream
        hidden_states.record_stream(stream)
        with torch.cuda.device(self.owner):
            with torch.cuda.stream(stream):
                output = shared._layer(hidden_states)
                stable = _shared_output_slot(shared, index, output)
                if output is not stable:
                    stable.copy_(output)
        pending["input"] = None
        pending["output"] = stable
        _COUNTS["shared_overlap_defers"] += 1
        if self._diagnostics is not None:
            self._count_diagnostic("shared_overlap", "deferred_calls")
        return shared

    def run(self, layer, x, weights, ids, original):
        if _ROUND_RECORD_FUNCTIONS:
            with torch.profiler.record_function("qwen38.TransferArena.run"):
                return self._run_transfer(layer, x, weights, ids, original)
        return self._run_transfer(layer, x, weights, ids, original)

    def _run_transfer(self, layer, x, weights, ids, original):
        collecting = _ROUTE_CALIBRATION and x.shape[0] <= 8
        asynchronous = self.async_transfers and not collecting
        started = time.perf_counter() if collecting else 0
        with torch.cuda.device(self.owner):
            self.host_x.copy_(x, non_blocking=True)
            self.host_weights.copy_(weights, non_blocking=True)
            self.host_ids.copy_(ids, non_blocking=True)
            if self._diagnostics is not None:
                self._record_diagnostic_copy("owner_gpu_to_pinned_host_input", calls=3)
            if asynchronous:
                self.owner_ready.record(torch.cuda.current_stream(self.owner))
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "owner_ready_records")
            else:
                torch.cuda.current_stream(self.owner).synchronize()
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "owner_stream_synchronize_calls")
        record = (_record_routes(layer, self.host_ids, time.perf_counter() - started)
                  if collecting else None)
        helper_started = time.perf_counter() if collecting else 0
        deferred_shared = None
        with torch.cuda.device(self.helper):
            if asynchronous:
                torch.cuda.current_stream(self.helper).wait_event(self.owner_ready)
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "helper_wait_owner_event_calls")
            if not self.io_graphs:
                self.remote_x.copy_(self.host_x, non_blocking=True)
                self.remote_weights.copy_(self.host_weights, non_blocking=True)
                self.remote_ids.copy_(self.host_ids, non_blocking=True)
                if self._diagnostics is not None:
                    self._record_diagnostic_copy("pinned_host_to_helper_gpu_input", calls=3)
            value = self._expert_value(layer, original)
            if not self.io_graphs:
                self.host_y.copy_(value, non_blocking=True)
                if self._diagnostics is not None:
                    self._record_diagnostic_copy("helper_gpu_to_pinned_host_output")
            deferred_shared = self._run_deferred_shared(layer)
            if asynchronous:
                self.helper_ready.record(torch.cuda.current_stream(self.helper))
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "helper_ready_records")
            else:
                torch.cuda.current_stream(self.helper).synchronize()
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "helper_stream_synchronize_calls")
        if record is not None:
            record["helper_step_s"] += time.perf_counter() - helper_started
        with torch.cuda.device(self.owner):
            if asynchronous:
                torch.cuda.current_stream(self.owner).wait_event(self.helper_ready)
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "owner_wait_helper_event_calls")
            self.returned.copy_(self.host_y, non_blocking=True)
            if self._diagnostics is not None:
                self._record_diagnostic_copy("pinned_host_to_owner_gpu_output")
            if deferred_shared is not None:
                torch.cuda.current_stream(self.owner).wait_stream(deferred_shared._stream)
                if self._diagnostics is not None:
                    self._count_diagnostic("event_ops", "owner_wait_shared_stream_calls")
        if asynchronous:
            _COUNTS["async_event_bridges"] += 1
        _COUNTS["remote_calls"] += 1
        _COUNTS["activation_bytes"] += sum(t.numel() * t.element_size()
            for t in (x, weights, ids, self.returned))
        return self.returned




def install():
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.layers.fused_moe import modular_kernel
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.v1.worker.workspace import WorkspaceManager
    if getattr(BaseModelLoader, "_qwen38_peer_installed", False):
        return
    if _QAD_NATIVE:
        from qad_csf_loader import iter_qad_model_weights
        from qad_nvfp4_backend import (
            assert_transfer_arena_compatible, install_target_method_selector,
            is_native_nvfp4_method, nvfp4_runtime_status,
            rebind_nvfp4_after_to, restore_target_method_selector,
        )
        from qad_ple_mmap import install_qad_ple_mmap
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
        install_qad_ple_mmap()
        from vllm.model_executor.model_loader import utils as loader_utils
        original_device_loading_context = loader_utils.device_loading_context

        @contextmanager
        def native_source_loading_context(module, target_device):
            if is_native_nvfp4_method(getattr(module, "quant_method", None)):
                # This method packs original CPU planes directly into helper
                # storage; moving raw planes first also restores new packed
                # parameters back to CPU in the generic offload context.
                yield module
            else:
                with original_device_loading_context(module, target_device) as loaded:
                    yield loaded

        loader_utils.device_loading_context = native_source_loading_context
    from device_stream_cache import install_device_stream_cache
    install_device_stream_cache()
    if os.environ.get("QWEN38_QSA_STABLE_TOPK") == "1":
        from qsa_stable_topk import install_qsa_stable_topk
        _NUMERIC_ADAPTERS["qsa"] = install_qsa_stable_topk()
    if os.environ.get("QWEN38_MOE_STABLE_ALIGN") == "1":
        from deterministic_moe_align import install_deterministic_marlin_alignment
        _, scope = install_deterministic_marlin_alignment()
        _NUMERIC_ADAPTERS["moe_alignment"] = {"scope": scope}
    original_shared_forward = SharedExperts.forward
    original_shared_sync = SharedExperts.maybe_sync_shared_experts_stream

    def stable_shared_forward(module, hidden_states, order):
        pending = getattr(module, "_qwen38_shared_deferred", None)
        if pending is not None and getattr(order, "name", None) == "MULTI_STREAM_OVERLAPPED":
            expected = module._determine_shared_experts_order(hidden_states)
            index = module._output_idx
            if (type(order) is type(expected) and order == expected
                    and pending["index"] == index and pending["input"] is None
                    and isinstance(pending["output"], torch.Tensor)):
                assert module._output[index] is None
                module._output[index] = pending["output"]
                module._qwen38_shared_deferred = None
                return None
            module._qwen38_shared_deferred = None
        result = original_shared_forward(module, hidden_states, order)
        if (not is_forward_context_available()
                or get_forward_context().cudagraph_runtime_mode.name == "NONE"):
            return result
        index = module._output_idx
        output = module._output[index]
        if output is None:
            return result
        # Every opaque shared-expert branch can feed a piecewise graph,
        # including the draft's MK-internal branch that bypasses aux streams.
        stable = _shared_output_slot(module, index, output)
        if output is not stable:
            stable.copy_(output)
        module._output[index] = stable
        return result

    def deferred_shared_sync(module, hidden_states):
        if not _shared_overlap_eligible(module, hidden_states, owner):
            return original_shared_sync(module, hidden_states)
        index = module._output_idx
        if (module._output[index] is not None
                or getattr(module, "_qwen38_shared_deferred", None) is not None):
            module._qwen38_shared_deferred = None
            return original_shared_sync(module, hidden_states)
        result = original_shared_sync(module, hidden_states)
        module._qwen38_shared_deferred = {
            "index": index, "input": hidden_states, "output": None}
        return result

    SharedExperts.forward = stable_shared_forward
    SharedExperts.maybe_sync_shared_experts_stream = deferred_shared_sync
    original_iterator = DefaultModelLoader._get_weights_iterator
    def selected_iterator(self, source):
        folder = Path(source.model_or_path)
        if _QAD_NATIVE:
            if source.subfolder is not None:
                folder /= source.subfolder
            return iter_qad_model_weights(folder)
        values = original_iterator(self, source)
        if folder.name != "diagnostic-four-layers":
            return values
        allowed = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        return ((name, value) for name, value in values
                if name.removeprefix(source.prefix) in allowed)
    DefaultModelLoader._get_weights_iterator = selected_iterator
    original_load = BaseModelLoader.load_model
    original_forward = RoutedExperts.forward_modular
    owner, helper = 0, 1
    original_workspace = modular_kernel.current_workspace_manager
    helper_workspaces = {}
    native_workspace_pool = {}
    qad_hot_runtime = None

    def device_workspace():
        if torch.cuda.current_device() != helper:
            return original_workspace()
        if helper not in helper_workspaces:
            base = original_workspace()
            helper_workspaces[helper] = WorkspaceManager(
                torch.device("cuda", helper), base._num_ubatches, base._num_lanes)
        return helper_workspaces[helper]

    # The upstream singleton allocates modular MoE outputs on the owner's GPU.
    modular_kernel.current_workspace_manager = device_workspace
    local_layer_count = int(os.environ.get("QWEN38_LOCAL_EXPERT_LAYERS", "0"))
    if local_layer_count < 0 or local_layer_count > 12:
        raise ValueError("RTX3090 trial permits 0 through 12 local expert layers")
    if _QAD_NATIVE and local_layer_count:
        raise ValueError("Native QAD keeps all 48 complete expert banks on the GB10 helper")
    hot_profile_path = os.environ.get("QWEN38_HOT_CACHE_PROFILE")
    hot_profile = None
    if hot_profile_path:
        if local_layer_count or _SHARED_OVERLAP or os.environ.get("QWEN38_ASYNC_TRANSFERS") != "1":
            raise ValueError("Hot-cache trial requires zero local layers, no shared overlap and async transfers")
        hot_profile = json.loads(Path(hot_profile_path).read_text())
        if (hot_profile.get("scope") != "past calibration fit; not heldout or TPS"
                or hot_profile.get("layer_count") != 48
                or len(hot_profile.get("layers", {})) != 48):
            raise ValueError("Hot-cache trial requires the complete 48-layer past-fit profile")
        if not _QAD_NATIVE:
            from hot_expert_cache import HotExpertCache
            from hot_cache_bridge import HotCacheBridge

    def _apply_tail_cutoff(topk_weights, topk_ids):
        """Router pruning: optional hard top-k limit, then weight-threshold tail cutoff."""
        limit = int(os.environ.get("QWEN38_ROUTER_TOPK_LIMIT", "0") or 0)
        if 0 < limit < topk_weights.shape[-1]:
            keep_k = torch.zeros_like(topk_weights, dtype=torch.bool)
            idx = torch.topk(topk_weights, k=limit, dim=-1).indices
            keep_k.scatter_(-1, idx, True)
            topk_weights = topk_weights * keep_k.to(topk_weights.dtype)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            # keep ids intact: -1 sentinels break the verifier's draft comparison
            topk_ids = topk_ids.clone()
        cutoff = float(os.environ.get("QWEN38_ROUTER_TAIL_CUTOFF", "0") or 0)
        if cutoff <= 0 or topk_weights.shape[-1] <= 4:
            return topk_weights, topk_ids
        keep = topk_weights >= cutoff
        # keep at least the top 4 experts so the distribution cannot collapse
        forced = torch.topk(topk_weights, k=min(4, topk_weights.shape[-1]), dim=-1).indices
        keep.scatter_(-1, forced, True)
        masked_ids = topk_ids.masked_fill(~keep, -1)
        weights = topk_weights * keep.to(topk_weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights, masked_ids

    def remote_forward(self, x, topk_weights, topk_ids, shared_experts=None,
                       shared_experts_input=None):
        topk_weights, topk_ids = _apply_tail_cutoff(topk_weights, topk_ids)
        if not getattr(self, "_qwen38_remote", False):
            output = original_forward(self, x, topk_weights, topk_ids,
                                      shared_experts, shared_experts_input)
            if (not is_forward_context_available()
                    or get_forward_context().cudagraph_runtime_mode.name == "NONE"):
                return output
            # The local MTP also inherits whole-MoE graph splits. Modular
            # kernels allocate a fresh output, so its consumer needs ownership.
            outputs = getattr(self, "_qwen38_local_routed_outputs", None)
            if outputs is None:
                outputs = self._qwen38_local_routed_outputs = {}
            key = (tuple(output.shape), output.dtype, output.device)
            stable = outputs.get(key)
            if stable is None:
                stable = outputs[key] = torch.empty_like(output)
            stable.copy_(output)
            return stable
        # Shared-expert execution remains in the separate vLLM MoERunner path.
        if _QAD_NATIVE and is_native_nvfp4_method(self.quant_method):
            assert_transfer_arena_compatible(self)
        elif self.quant_method.moe_kernel.impl.prepare_finalize.supports_async():
            raise RuntimeError("Peer experiment requires synchronous MoE finalization")
        if x.device.index != owner or self.w13_weight.device.index != helper:
            raise RuntimeError("Peer expert device ownership mismatch")
        key = (tuple(x.shape), x.dtype, tuple(topk_ids.shape), topk_ids.dtype,
               topk_weights.dtype)
        arena = _ARENAS.get(key)
        if arena is None:
            arena = TransferArena(x, topk_weights, topk_ids, owner, helper)
            _ARENAS[key] = arena
        if _QAD_NATIVE:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_router import _get_padding_mask
            x, topk_weights, topk_ids = arena.qad_active_inputs(
                x, topk_weights, topk_ids, _get_padding_mask(x.shape[0]),
            )
        try:
            hot_cache = _HOT_CACHES.get(self)
            if (hot_cache is not None
                    and 1 <= x.shape[0] <= (qad_hot_runtime.max_rows if _QAD_NATIVE else 8)
                    and (not _QAD_NATIVE or _QAD_HOT_CACHE_ENABLED)):
                if _QAD_NATIVE:
                    return qad_hot_runtime.run_small(self, arena, x, topk_weights, topk_ids, _COUNTS)
                bridge = getattr(arena, "hot_bridge", None)
                if bridge is None:
                    bridge = arena.hot_bridge = HotCacheBridge(arena, _COUNTS)
                return bridge.run(self, hot_cache, x, topk_weights, topk_ids)
            return arena.run(self, x, topk_weights, topk_ids, original_forward)
        except Exception:
            shared = getattr(self, "_qwen38_shared_experts", None)
            if shared is not None:
                shared._qwen38_shared_deferred = None
            raise

    def load_model(self, vllm_config, model_config, prefix=""):
        nonlocal qad_hot_runtime
        if torch.cuda.get_device_name(owner) != "NVIDIA GeForce RTX 3090":
            raise RuntimeError("CUDA0 must be the RTX3090 for this experiment")
        if torch.cuda.get_device_name(helper) != "NVIDIA GB10":
            raise RuntimeError("CUDA1 must be the GB10 for this experiment")
        original_device = vllm_config.load_config.device
        draft_request = _QAD_NATIVE and "Qwen3_8FlashNextMTP" in model_config.architectures
        load_device = owner if draft_request else helper
        vllm_config.load_config.device = f"cuda:{load_device}"
        if _QAD_NATIVE and not draft_request:
            before_pack = None
            if hot_profile is not None:
                from qad_hot_expert_cache import QADHotCacheRuntime
                qad_hot_runtime = QADHotCacheRuntime(hot_profile_path, vllm_config, owner=owner, helper=helper)
                def capture_hot_experts(layer, source_prefix):
                    match = re.search(r"\.layers\.(\d+)\.mlp\.experts$", source_prefix)
                    if match is None:
                        raise RuntimeError(f"QAD cache source layer prefix is not canonical: {source_prefix}")
                    _HOT_CACHES[layer] = qad_hot_runtime.add_layer(layer, int(match.group(1)))
                before_pack = capture_hot_experts
            install_target_method_selector(helper_device=helper, before_pack=before_pack)
        try:
            with torch.cuda.device(load_device):
                model = original_load(self, vllm_config, model_config, prefix)
        finally:
            vllm_config.load_config.device = original_device
            if _QAD_NATIVE and not draft_request:
                restore_target_method_selector()
        is_draft = "MTP" in type(model).__name__ or "Predictor" in type(model).__name__
        if _QAD_NATIVE and is_draft != draft_request:
            raise RuntimeError("QAD model architecture and loaded MTP ownership disagree")
        kept = 0
        hot_owner_bytes = 0
        hot_helper_bytes = 0
        hot_layer_names = set()
        def move(module, name="", parent=None):
            nonlocal kept, hot_owner_bytes, hot_helper_bytes
            if isinstance(module, RoutedExperts):
                local = is_draft or kept < local_layer_count
                if not is_draft and local:
                    kept += 1
                device = owner if local else helper
                with torch.cuda.device(device):
                    module.quant_method.moe.device = torch.device("cuda", device)
                    module.to(device)
                    if _QAD_NATIVE and is_native_nvfp4_method(module.quant_method):
                        if local or is_draft:
                            raise RuntimeError("Native QAD target experts must remain helper-owned")
                        rebind_nvfp4_after_to(module, helper)
                        assert_transfer_arena_compatible(module)
                        # TransferArena consumes each helper result before the next
                        # layer; share shape/stream scratch, never packed weights.
                        module._qad_nvfp4_workspace_pool = native_workspace_pool
                    elif _QAD_NATIVE and isinstance(module.quant_method, ModelOptNvFp4FusedMoE):
                        if not is_draft or not local or not module.quant_method.use_a16:
                            raise RuntimeError("Only the original owner-loaded W4A16 QAD MTP may use ModelOpt Marlin")
                        if module.w13_weight.device.index != owner or module.w2_weight.device.index != owner:
                            raise RuntimeError("QAD MTP was not packed directly on the RTX3090")
                    elif isinstance(module.quant_method, AutoGPTQMoEMethod):
                        module.quant_method._setup_kernel(module)
                    elif not isinstance(module.quant_method, UnquantizedFusedMoEMethod):
                        raise TypeError("Unsupported expert kernel in the pinned benchmark image")
                module._qwen38_remote = not local
                module._qwen38_route_name = name
                if hot_profile is not None and not local and not is_draft:
                    if _QAD_NATIVE:
                        hot_cache = qad_hot_runtime.cache_for(module)
                        hot_owner_bytes = qad_hot_runtime.owner_persistent_bytes
                        hot_helper_bytes = qad_hot_runtime.helper_persistent_bytes
                    else:
                        hot_cache = HotExpertCache(module, hot_profile["layers"][name], owner, helper)
                        hot_owner_bytes += hot_cache.owner_bytes
                        hot_helper_bytes += hot_cache.helper_persistent_bytes
                    if hot_owner_bytes > _HOT_CACHE_OWNER_BUDGET:
                        raise RuntimeError(f"Hot-cache owner budget exceeded: {hot_owner_bytes}")
                    _HOT_CACHES[module] = hot_cache
                    hot_layer_names.add(name)
                if not local and not is_draft and parent is not None:
                    shared_siblings = [
                        child for child in parent.children()
                        if isinstance(child, SharedExperts)]
                    if len(shared_siblings) == 1:
                        object.__setattr__(module, "_qwen38_shared_experts", shared_siblings[0])
                        object.__setattr__(shared_siblings[0], "_qwen38_remote_routed", module)
                        shared_siblings[0]._qwen38_shared_deferred = None
                _PLACEMENT.append({"module": name, "draft": is_draft,
                    "device": device, "class": type(module.quant_method).__name__,
                    "bytes": sum(p.numel() * p.element_size() for p in module.parameters())})
                if _QAD_NATIVE and is_native_nvfp4_method(module.quant_method):
                    _PLACEMENT[-1]["native_nvfp4"] = nvfp4_runtime_status(module)
                return
            for child_name, child in module.named_children():
                move(child, name + "." + child_name, module)
            _move_trunk_state(module, owner)
        move(model, prefix or type(model).__name__)
        torch.cuda.synchronize(owner)
        torch.cuda.synchronize(helper)
        with torch.cuda.device(owner):
            torch.cuda.empty_cache()
        with torch.cuda.device(helper):
            torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated(owner)
        if not is_draft and used > 21 * 1024**3:
            raise RuntimeError(f"RTX3090 target weights exceed 21GiB safety budget: {used}")
        print({"peer_model_placed": type(model).__name__, "primary_bytes": used,
               "helper_bytes": torch.cuda.memory_allocated(helper),
               "local_target_expert_layers": kept}, flush=True)
        if hot_profile is not None and not is_draft:
            if (_QAD_NATIVE and len(hot_layer_names) != 48) or (
                    not _QAD_NATIVE and hot_layer_names != set(hot_profile["layers"])):
                raise RuntimeError("Loaded target layers do not match the complete hot-cache profile")
            _NUMERIC_ADAPTERS["hot_cache"] = {
                "scope": hot_profile["scope"], "source": hot_profile["source"],
                "per_layer": hot_profile["per_layer"], "layers": len(hot_layer_names),
                "owner_budget_bytes": _HOT_CACHE_OWNER_BUDGET,
                "owner_bytes": hot_owner_bytes, "helper_persistent_bytes": hot_helper_bytes,
                "owner_allocation_plan": (
                    qad_hot_runtime.owner_allocation_plan if _QAD_NATIVE else None),
                "numerical_mode": "cross-device difference candidate; tensor equivalence unproved"}
            print({"peer_hot_cache_placed": _NUMERIC_ADAPTERS["hot_cache"]}, flush=True)
        return model

    RoutedExperts.forward_modular = remote_forward
    BaseModelLoader.load_model = load_model
    BaseModelLoader._qwen38_peer_installed = True


install()


class PeerWorkerExtension(BenchmarkWorkerExtension):
    def peer_inventory(self):
        numeric = dict(_NUMERIC_ADAPTERS)
        if "qsa" in numeric:
            from qsa_stable_topk import qsa_stable_topk_state
            numeric["qsa"] = qsa_stable_topk_state()
        if "moe_alignment" in numeric:
            from deterministic_moe_align import alignment_state
            numeric["moe_alignment"] = alignment_state()
        target_graph_mode = None
        vllm_config = getattr(getattr(self, "model_runner", None), "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        configured_mode = getattr(compilation_config, "cudagraph_mode", None)
        if configured_mode is not None:
            target_graph_mode = getattr(configured_mode, "name", str(configured_mode))
        transport_diagnostics = {
            "enabled": _PEER_DIAGNOSTICS,
            "record_function_enabled": _ROUND_RECORD_FUNCTIONS,
            "target_cudagraph_mode": target_graph_mode,
            "helper_graph_scope": (
                "TransferArena QWEN38_HELPER_GRAPHS; separate from target_cudagraph_mode"),
            "helper_transfer_arenas": (
                [snapshot for arena in _ARENAS.values()
                 if (snapshot := arena._diagnostic_snapshot()) is not None]
                if _PEER_DIAGNOSTICS else []),
        }
        async_env_enabled = os.environ.get("QWEN38_ASYNC_TRANSFERS") == "1"
        async_arena_states = [
            {
                "owner": arena.owner,
                "helper": arena.helper,
                "async_transfers": bool(arena.async_transfers),
            }
            for arena in _ARENAS.values()
        ]
        runtime_controls = {
            "peer_async_transfers": {
                "environment_enabled": async_env_enabled,
                "arena_states": async_arena_states,
                "consistent": bool(async_arena_states) and all(
                    state["async_transfers"] == async_env_enabled
                    for state in async_arena_states
                ),
            },
            "peer_shared_overlap": {
                "runtime_enabled": bool(_SHARED_OVERLAP),
                "environment_enabled": (
                    os.environ.get("QWEN38_SHARED_OVERLAP") == "1"
                ),
                "consistent": (
                    bool(_SHARED_OVERLAP)
                    == (os.environ.get("QWEN38_SHARED_OVERLAP") == "1")
                ),
            },
        }
        return {"placement": _PLACEMENT, "counters": dict(_COUNTS),
                "allocated": [torch.cuda.memory_allocated(i) for i in (0, 1)],
                "numeric_adapters": numeric,
                "transport_diagnostics": transport_diagnostics,
                "runtime_controls": runtime_controls}

    def peer_shared_overlap(self, enabled=False):
        global _SHARED_OVERLAP
        _SHARED_OVERLAP = bool(enabled)
        os.environ["QWEN38_SHARED_OVERLAP"] = str(int(_SHARED_OVERLAP))
        return {"enabled": _SHARED_OVERLAP,
                "deferred_calls": _COUNTS["shared_overlap_defers"]}

    def peer_async_transfers(self, enabled=False):
        enabled = bool(enabled)
        os.environ["QWEN38_ASYNC_TRANSFERS"] = str(int(enabled))
        for arena in _ARENAS.values():
            if enabled and arena.owner_ready is None:
                arena.owner_ready = torch.cuda.Event()
                arena.helper_ready = torch.cuda.Event()
            arena.async_transfers = enabled
        return {"enabled": enabled, "arenas": len(_ARENAS),
                "calibration_uses_synchronous_bridge": True}

    def peer_route_calibration(self, enabled=False, reset=False):
        global _ROUTE_CALIBRATION, _ROUTES
        _ROUTE_CALIBRATION = bool(enabled)
        if reset:
            _ROUTES = {}
        return {"collecting": _ROUTE_CALIBRATION, "layers": _ROUTES}

    def peer_event_model_smoke(self):
        from event_model_smoke import run
        return run(self)

    def peer_hot_cache_smoke(self):
        import faulthandler
        from hot_cache_smoke import run
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            return run(self)
        finally:
            faulthandler.cancel_dump_traceback_later()

    def peer_check_finite(self):
        from torch.utils._pytree import tree_leaves
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
        original_graph_call = CUDAGraphWrapper.__call__
        def checked_graph_call(wrapper, *args, **kwargs):
            wrapper.is_debugging_mode = True
            try:
                return original_graph_call(wrapper, *args, **kwargs)
            except AssertionError:
                print("PEER_GRAPH_INPUTS " + json.dumps({
                    "inputs": [{"shape": list(value.shape), "dtype": str(value.dtype),
                                "device": str(value.device), "pointer": value.data_ptr()}
                               for value in args if isinstance(value, torch.Tensor)],
                    "latest_shared": getattr(self, "_peer_latest_shared", None),
                    "routed_outputs": [{"key": str(key), "pointer": arena.returned.data_ptr()}
                                       for key, arena in _ARENAS.items()]}), flush=True)
                raise
        CUDAGraphWrapper.__call__ = checked_graph_call

        checked_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        def make_hook(name):
            def check(module, inputs, output):
                if isinstance(module, SharedExperts):
                    value = module._output[module._output_idx]
                    if value is not None:
                        self._peer_latest_shared = {"module": name, "shape": list(value.shape),
                                                    "pointer": value.data_ptr()}
                for value in tree_leaves(output):
                    if (isinstance(value, torch.Tensor) and value.dtype in checked_dtypes
                            and not bool(torch.isfinite(value).all().item())):
                        input_health = [
                            {"shape": list(item.shape), "finite": bool(torch.isfinite(item).all().item())}
                            for item in tree_leaves(inputs)
                            if isinstance(item, torch.Tensor) and item.dtype in checked_dtypes]
                        raise RuntimeError(
                            f"Nonfinite peer output at {name} ({type(module).__name__}), "
                            f"shape={tuple(value.shape)}, device={value.device}, inputs={input_health}")
            return check

        def wrap_boundary(module, name, method_name):
            original = getattr(module, method_name)
            check = make_hook(name + "." + method_name)
            def checked(*args, **kwargs):
                if method_name == "forward_modular":
                    check(module, (), args)
                result = original(*args, **kwargs)
                check(module, args, (args, kwargs, result))
                return result
            setattr(module, method_name, checked)

        self._peer_finite_hooks = []
        boundaries = 0
        for name, module in self.model_runner.model.named_modules():
            self._peer_finite_hooks.append(module.register_forward_hook(make_hook(name)))
            for method_name in ("forward_modular", "_forward_core", "_forward_core_fused_norm_packed"):
                if callable(getattr(module, method_name, None)):
                    wrap_boundary(module, name, method_name)
                    boundaries += 1
        return {"checked_modules": len(self._peer_finite_hooks),
                "opaque_boundaries": boundaries, "stage": "real_requests_only"}
