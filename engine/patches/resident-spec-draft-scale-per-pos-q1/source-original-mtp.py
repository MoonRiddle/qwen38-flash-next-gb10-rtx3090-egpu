# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.8-Flash-Next MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen3.8-Flash-Next backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_8_flash_next import (
    Qwen3_8FlashNextTextConfig,
)

from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen38next_low_latency_gemm
from .model import (
    _HC_WEIGHTS_MAPPER,
    _QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES,
    Qwen3_8FlashNextDecoderLayer,
    Qwen3_8FlashNextMixtureOfExperts,
)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen3.8-Flash-Next checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen3_8FlashNextMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_8FlashNextMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen3_8FlashNextTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen3_8FlashNextDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size

        if get_pp_group().is_first_rank:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = self.fc_hidden(hidden_states)
            # Add the embedding residual to every branch, then fold back
            # to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.
            hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
            hidden_states = hidden_states.flatten(-2)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            # As in the target model, PP carries a materialized tensor rather
            # than the delayed hidden/output/injection tuple.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_8FlashNextMTP(nn.Module, SupportsPP, Qwen3_8FlashNextMixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_8FlashNextMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_8FlashNextMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=vllm_config.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen38next_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(remap_weight_names())


__all__ = ["Qwen3_8FlashNextMTP", "Qwen3_8FlashNextMultiTokenPredictor"]

# --- qwen38-flash-dgx: private MTP draft head (VLLM_MTP_DRAFT_VOCAB=<ids.npy>, VLLM_MTP_DRAFT_HEAD=int4)
import os as _dv_os
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)
_DV_CHUNK = 16384


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    if getattr(self, "_resident_use_bf16_prose_head", False):
        st = getattr(self, "_resident_prose_bf16_state", None)
        if st is None:
            raise RuntimeError("BF16 prose draft head state is not prepared")
    else:
        st = getattr(self, "_dv_state", None)
        if st is None:
            st = self._dv_state = _dv_build(self, hidden_states)
    ids, head, vocab = st
    red = head(hidden_states)
    if getattr(self, "_resident_apply_draft_scale", False):
        red.mul_(self.logits_processor.scale)
    if ids is None:
        return red[:, :vocab] if red.shape[1] != vocab else red
    full = red.new_full((red.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, red)
    return full


def _dv_validate_private_head_state(self, hidden_states, state, expected_mode):
    if not isinstance(state, tuple) or len(state) != 3:
        raise RuntimeError("private MTP draft head state must be (ids, apply, vocab)")
    ids, apply, vocab = state
    if not callable(apply):
        raise RuntimeError("private MTP draft head state has no callable head")
    mode = getattr(apply, "_dv_head_mode", None)
    if mode not in ("int4", "bf16") or (
        expected_mode is not None and mode != expected_mode
    ):
        raise RuntimeError("private MTP draft head state has an unexpected mode")
    if type(vocab) is not int or vocab <= 0:
        raise RuntimeError("private MTP draft head state has invalid vocabulary geometry")
    if hidden_states.ndim != 2:
        raise RuntimeError("private MTP draft hidden states must be rank two")
    input_features = getattr(apply, "_dv_input_features", None)
    output_features = getattr(apply, "_dv_output_features", None)
    device = getattr(apply, "_dv_device", None)
    if type(input_features) is not int or type(output_features) is not int:
        raise RuntimeError("private MTP draft head state has incomplete geometry metadata")
    if hidden_states.shape[1] != input_features or hidden_states.device != device:
        raise RuntimeError("private MTP draft head input geometry does not match prepared state")
    if output_features <= 0:
        raise RuntimeError("private MTP draft head state has invalid output geometry")
    if ids is None:
        if output_features < vocab:
            raise RuntimeError("private MTP draft head state is narrower than the vocabulary")
    else:
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim != 1
            or ids.dtype != torch.int64
            or ids.device != hidden_states.device
            or ids.numel() != output_features
        ):
            raise RuntimeError("private MTP draft vocabulary indices have invalid geometry")


def _dv_validate_prose_bf16_state(self, hidden_states, state):
    width = getattr(state[1], "_dv_input_features", None)
    hc_count = getattr(getattr(self, "model", None), "hc_count", None)
    if type(width) is not int or type(hc_count) is not int or hc_count <= 0:
        raise RuntimeError("MTP HC geometry is unavailable for prepared BF16 head")
    if hidden_states.ndim != 2 or hidden_states.shape[1] not in (width, hc_count * width):
        raise RuntimeError("MTP preparation input is neither single-stream nor the configured HC width")
    _dv_validate_private_head_state(self, hidden_states[:1, :width], state, "bf16")


def _dv_build(self, hidden_states, *, head_mode=None):
    if head_mode is not None and (
        type(head_mode) is not str or head_mode not in ("int4", "bf16")
    ):
        raise ValueError("head_mode must be None, 'int4', or 'bf16'")
    import time as _time
    import numpy as _np

    t0 = _time.time()
    dev = hidden_states.device
    head = self.lm_head  # the target's lm_head, shared in by the proposer
    w = getattr(head, "weight", None)
    dense = w is not None and w.dim() == 2
    vocab = int(getattr(self.logits_processor, "org_vocab_size", 0)
                or getattr(self.logits_processor, "vocab_size", 0)
                or (w.shape[0] if dense else head.org_vocab_size))
    if dense:
        out_f, in_f = w.shape
        rows = lambda sel: w.index_select(0, torch.as_tensor(sel, device=w.device)).to(torch.bfloat16)
        full_bytes = w.numel() * w.element_size()
    else:
        rows, out_f, in_f, full_bytes = _dv_gptq_rows()

    ids_path = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB")
    int4 = (
        head_mode == "int4"
        if head_mode is not None
        else _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD", "").lower() == "int4"
    )
    if ids_path:
        ids = _np.load(ids_path).astype(_np.int64)
        ids = ids[(ids >= 0) & (ids < vocab)]
        if int4 and len(ids) % 64:  # Marlin needs N % 64 == 0; duplicates write identical values
            ids = _np.concatenate([ids, _np.repeat(ids[-1], 64 - len(ids) % 64)])
        sel = ids
    else:
        sel = _np.arange(out_f)
        ids = None
    n = len(sel)

    if int4:
        from vllm.scalar_type import scalar_types as _st
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize as _mq
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new as _mws, apply_gptq_marlin_linear as _mapply)
        qws, scs = [], []
        for i in range(0, n, _DV_CHUNK):
            wt = rows(sel[i:i + _DV_CHUNK]).to(dev).t().contiguous()      # [in, chunk] bf16
            _, qw, s, _, _, _ = _mq(wt, _st.uint4b8, 128, False)
            qws.append(qw); scs.append(s)
            del wt
        qw = torch.cat(qws, 1).contiguous(); sc = torch.cat(scs, 1).contiguous()
        del qws, scs
        ws = _mws(dev)
        e = torch.empty(0, dtype=torch.int, device=dev)

        def apply(h, qw=qw, sc=sc, ws=ws, e=e, n=n, k=in_f):
            return _mapply(h.to(sc.dtype), qw, sc, e, e, e, ws, _st.uint4b8, n, k, True)
        bytes_ = qw.numel() * 4 + sc.numel() * sc.element_size()
        kind = "int4 g128 RTN GPTQ-Marlin"
    else:
        wk = torch.cat([rows(sel[i:i + _DV_CHUNK]).to(dev) for i in range(0, n, _DV_CHUNK)]).contiguous()

        def apply(h, wk=wk):
            return torch.nn.functional.linear(h.to(wk.dtype), wk)
        bytes_ = wk.numel() * wk.element_size()
        kind = "bf16"

    apply._dv_head_mode = "int4" if int4 else "bf16"
    apply._dv_input_features = in_f
    apply._dv_output_features = n
    apply._dv_device = dev
    apply._dv_weight_bytes = bytes_

    # self-check on the first real hidden states: private head vs the shared target head
    try:
        ref = self.logits_processor(self.lm_head, hidden_states)
        if ref is not None:
            ref = ref[:, :vocab].float()
            got = apply(hidden_states)[:, :ids.shape[0] if ids is not None else vocab].float()
            if ids is not None:
                ref = ref[:, torch.as_tensor(ids, device=ref.device)]
            else:
                got = got[:, :vocab]
            top1 = (ref.argmax(1) == got.argmax(1)).float().mean().item()
            rel = ((got - ref).norm() / ref.norm()).item()
            _dv_logger.info("MTP private draft head self-check: top-1 agreement %.3f, rel err %.4f (T=%d)",
                            top1, rel, ref.shape[0])
    except Exception as ex:  # never let a diagnostic kill the drafter
        _dv_logger.warning("MTP private draft head self-check skipped: %r", ex)

    _dv_logger.info("MTP private draft head: %s, %s of %d rows, %.0f -> %.0f MiB per draft step, built in %.1fs",
                    kind, ("%d" % ids.shape[0]) if ids is not None else "all", vocab,
                    full_bytes / 2**20, bytes_ / 2**20, _time.time() - t0)
    ids_t = torch.as_tensor(ids, device=dev) if ids is not None else None
    return ids_t, apply, vocab


def _dv_gptq_rows():
    """Row dequantizer for the checkpoint's GPTQ lm_head (int8/int4 v1 layout: qweight [in/pack, out]
    packed along in, qzeros stores zp-1, scales [in/group, out]). rows(sel) -> bf16 [len(sel), in]."""
    import json as _json
    import numpy as _np
    from safetensors import safe_open as _safe_open

    ckpt = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB_CKPT", "/model")
    wm = _json.load(open(_dv_os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    names = ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales")
    ts = {}
    for f in {wm[n] for n in names}:
        with _safe_open(_dv_os.path.join(ckpt, f), "pt", device="cpu") as sf:
            for n in names:
                if n in sf.keys():
                    ts[n] = sf.get_tensor(n)
    qw, qz, sc = (ts[n] for n in names)
    groups, out_f = sc.shape
    pack = out_f // qz.shape[1]
    bits = 32 // pack
    in_f = qw.shape[0] * pack
    group = in_f // groups
    mask = (1 << bits) - 1
    qwn = qw.numpy().view(_np.uint32)
    zp = int(qz.numpy().view(_np.uint32)[0, 0] & mask) + 1
    scn = sc.to(torch.float32).numpy()

    def rows(sel):
        q = qwn[:, sel]                                                  # [in/pack, k]
        nib = _np.stack([(q >> (i * bits)) & mask for i in range(pack)], axis=1).reshape(in_f, len(sel))
        wf = (nib.astype(_np.float32) - zp).reshape(groups, group, len(sel)) * scn[:, None, sel]
        return torch.from_numpy(_np.ascontiguousarray(wf.reshape(in_f, len(sel)).T)).to(torch.bfloat16)
    return rows, out_f, in_f, qw.numel() * 4 + sc.numel() * sc.element_size()


if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB") or _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD"):
    Qwen3_8FlashNextMTP.compute_logits = _dv_compute_logits  # type: ignore[method-assign]
    Qwen3_8FlashNextMTP._resident_use_bf16_prose_head = False
    Qwen3_8FlashNextMTP._resident_apply_draft_scale = False
    Qwen3_8FlashNextMTP._resident_prose_bf16_state = None
    Qwen3_8FlashNextMTP._resident_validate_prose_bf16_state = (
        _dv_validate_prose_bf16_state
    )
    _dv_logger.info("MTP private draft head enabled: vocab=%s head=%s",
                    _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"), _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD", "shared"))
