#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union, Iterable

import numpy as np

from mindspore import Parameter, Tensor, mint, nn, jit, mutable
from mindspore.common import dtype as mstype

from vllm_mindspore.model_executor.layers.logits_processor import \
    LogitsProcessor
from vllm_mindspore.model_executor.layers.sampler import (SamplerOutput,
                                                          get_sampler)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm_mindspore.model_executor.model_loader.weight_utils import \
    default_weight_loader
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix)
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata
from vllm_mindspore.model_executor.models.model_base import MsModelBase, Fake_Attention


from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import \
    QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.attention.backends.abstract import AttentionType
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.attention.backends.abstract import AttentionMetadata

import sys
# TODO: allow setting the path to mindone
# mindone_lib_path = '/home/hyx/vllm/mindone'
# sys.path.insert(0, mindone_lib_path)
from mindone.transformers.models.qwen2.modeling_qwen2_vllm import Qwen2Model 


class Qwen2ForCausalLM(MsModelBase):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config

        # NOTE: import mindone qwen2 here
        # the network
        print("D--: initialize Qwen2Model from mindone")
        self.model = Qwen2Model(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "model"))
        
        # the lm head
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              params_dtype=mstype.bfloat16,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(prefix, "lm_head"))
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.set_model_inputs()
        self.kv_caches = [Fake_Attention() for i in range(config.num_hidden_layers)]
        compilation_config = vllm_config.compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(config.num_hidden_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tuple[Tensor, Tensor]],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: IntermediateTensors = None,
        inputs_embeds: Tensor = None,
        **kwargs
    ) -> Union[Tensor, IntermediateTensors]:
        key_cache, value_cache = self.get_kvcache()
        if attn_metadata.num_prefill_tokens > 0:
            input_ids = input_ids.expand_dims(0)
        if attn_metadata.num_decode_tokens > 0:
            input_ids = input_ids.expand_dims(1)
        num_prefill_tokens = mutable(attn_metadata.num_prefill_tokens)
        num_decode_tokens = mutable(attn_metadata.num_decode_tokens)
        slot_mapping = attn_metadata.slot_mapping
        batch_valid_length = Tensor.from_numpy(np.array(attn_metadata.seq_lens, dtype=np.int32))
        q_seq_lens = Tensor.from_numpy(np.array(attn_metadata.query_lens, dtype=np.int32))
        block_tables = attn_metadata.block_tables

        # import pdb; pdb.set_trace()

        model_output = self.model(input_ids,
                                  positions,
                                  key_cache,
                                  value_cache,
                                  num_prefill_tokens,
                                  num_decode_tokens,
                                  slot_mapping,
                                  batch_valid_length,
                                  q_seq_lens,
                                  block_tables,
                                  intermediate_tensors=intermediate_tensors,
                                  inputs_embeds=inputs_embeds)
        if attn_metadata.num_prefill_tokens > 0:
            model_output = model_output.squeeze(0)
        if attn_metadata.num_decode_tokens > 0:
            model_output = model_output.squeeze(1)
        return model_output

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        params_dict = self.get_params_dict()
        self.model.load_weights(weights, params_dict)

    def sample(
        self, logits: Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits
