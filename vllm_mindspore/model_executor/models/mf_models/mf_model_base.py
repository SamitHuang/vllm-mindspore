#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from typing import Iterable, List, Optional, Set, Tuple, Union
import numpy as np

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.attention.backends.abstract import AttentionType
from vllm.logger import init_logger

import torch
import mindspore as ms
from mindspore import Tensor

from mindformers.tools.register.config import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from vllm_mindspore.model_executor.models.model_base import MsModelBase

logger = init_logger(__name__)


def _pad_to_max(x, max_len):
    return x + [-1] * (max_len - len(x))


def _pad_block_table(block_tables, seq_length, block_size, pad_size):
    # When prefill, the block_tables is a empty tensor.
    if len(block_tables.shape) < 2:
        fake_block_tables = ms.mint.empty(
            pad_size, seq_length // block_size, dtype=ms.int32, device="Ascend"
        )
        return fake_block_tables

    block_tables_list = block_tables.tolist()
    padded_block_tables = [
        _pad_to_max(block_table, seq_length // block_size)
        for block_table in block_tables_list
    ]

    return Tensor(np.array(padded_block_tables).astype(np.int32))


def _batch_seq(input_tokens, prefill):
    if prefill:
        return ms.ops.expand_dims(input_tokens, 0).to(ms.int32)

    return ms.mint.reshape(input_tokens, (-1, 1)).to(ms.int32)


class Fake_Attention:
    def __init__(self):
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config(
            ).parallel_config.pipeline_parallel_size)
        ]
        self.attn_type = AttentionType.DECODER


class MfModelBase(MsModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(MfModelBase, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        self.mf_config = MindFormerConfig(os.getenv("MINDFORMERS_MODEL_CONFIG"))
        build_context(self.mf_config, is_set_ms_ctx=False, is_init_ms=False)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = (
            self.mf_config.parallel_config
        )
        self.mf_config.model.model_config.parallel_config.model_parallel = (
            get_tensor_model_parallel_world_size()
        )
        self.mf_config.model.model_config.parallel_config.pipeline_stage = 1


    def update_mf_kvcaches(self):
        if self.mf_kvcaches_init:
            return

        forward_context = get_forward_context()
        for i in range(self.mf_model_config.num_layers):
            k_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            v_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][1]
            mf_k_cache, mf_v_cache = self.network.kvcache(i)
            mf_k_cache.set_device_address(
                k_cache._data_ptr(), k_cache.shape, k_cache.dtype
            )
            mf_v_cache.set_device_address(
                v_cache._data_ptr(), v_cache.shape, v_cache.dtype
            )
        self.mf_kvcaches_init = True


    def pad_block_table(self, block_tables, seq_length, block_size):
        raise NotImplementedError("pad_block_table not implemented.")


    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        self.update_mf_kvcaches()

        query_lens = attn_metadata.query_lens
        kv_cache_lens = attn_metadata.seq_lens_tensor.asnumpy() - query_lens
        if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max() == 0:
            is_prefill = True
        else:
            is_prefill = False

        q_seq_lens = ms.Tensor(query_lens, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(is_prefill, position_ids, query_lens)

        model_inputs = {}
        model_inputs["input_ids"] = _batch_seq(input_ids, is_prefill)
        model_inputs["batch_valid_length"] = ms.Tensor.from_numpy(np.expand_dims(
            attn_metadata.seq_lens_tensor.asnumpy(), 0))
        model_inputs["block_tables"] = self.pad_block_table(
            attn_metadata.block_tables,
            self.mf_model_config.seq_length,
            self.mf_model_config.block_size,
        )
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping
        model_inputs["position_ids"] = position_ids
        model_inputs["q_seq_lens"] = q_seq_lens
        model_inputs["attention_mask"] = attention_mask

        if is_prefill:
            self.network.phase = "prefill"
            if not self.set_flags:
                self.network.add_flags_custom(is_first_iteration=True)
            hidden_states = self.network(**model_inputs)
            self.network.phase = "increment"
            if not self.set_flags:
                self.network.add_flags_custom(is_first_iteration=False)
                self.set_flags = True
        else:
            hidden_states = self.network(**model_inputs)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        selected_token_indices = sampling_metadata.selected_token_indices
        if selected_token_indices is not None and selected_token_indices.numel() <= 0:
            logits = ms.mint.zeros((0, self.mf_model_config.vocab_size),
                                    dtype=self.mf_model_config.compute_dtype)
        else:
            hidden_states = hidden_states.index_select(0, selected_token_indices)
            logits = self.network.lm_head(hidden_states)
            logits = logits.reshape(-1, logits.shape[-1])

        return logits

    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        raise NotImplementedError("load_weight not implemented.")
