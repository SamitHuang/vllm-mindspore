# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
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
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Set, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import numpy as np
from transformers import OPTConfig
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm_mindspore.attention import Attention
from vllm_mindspore.model_executor.layers.activation import get_act_fn
from vllm_mindspore.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm_mindspore.model_executor.layers.logits_processor import LogitsProcessor
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm_mindspore.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm_mindspore.model_executor.models.model_base import Fake_Attention, MsModelBase
from vllm_mindspore.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata


class OPTLearnedPositionalEmbedding(mint.nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: ms.Type):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, dtype=dtype)

    def construct(self, positions: ms.Tensor):
        return super().construct(positions + self.offset)


class OPTAttention(nn.Cell):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            params_dtype=ms.float16,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            params_dtype=ms.float16,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.hard_mask = ms.Tensor([0], dtype=ms.float16).reshape(1, 1)

    def construct(
        self,
        hidden_states: ms.Tensor,
        key_cache: ms.Tensor,
        value_cache: ms.Tensor,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: ms.Tensor,
        batch_valid_length: Tuple[int],
        q_seq_lens: ms.Tensor,
        block_tables: ms.Tensor,
    ) -> ms.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(
            q,
            k,
            v,
            key_cache,
            value_cache,
            num_prefill_tokens,
            num_decode_tokens,
            slot_mapping,
            batch_valid_length,
            q_seq_lens,
            block_tables,
            None,
            self.hard_mask,
        )
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Cell):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = mint.nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine,
            dtype=ms.float16,
        )
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            params_dtype=ms.float16,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.activation_fn = get_act_fn(config.activation_function)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            params_dtype=ms.float16,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = mint.nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine,
            dtype=ms.float16,
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
        key_cache: ms.Tensor,
        value_cache: ms.Tensor,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: ms.Tensor,
        batch_valid_length: Tuple[int],
        q_seq_lens: ms.Tensor,
        block_tables: ms.Tensor,
    ) -> ms.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            key_cache,
            value_cache,
            num_prefill_tokens,
            num_decode_tokens,
            slot_mapping,
            batch_valid_length,
            q_seq_lens,
            block_tables,
        )
        hidden_states = mint.add(residual, hidden_states)
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = mint.add(residual, hidden_states)
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Cell):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            params_dtype=ms.float16,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size, dtype=ms.float16
        )

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                params_dtype=ms.float16,
                quant_config=quant_config,
                prefix=f"{prefix}.project_out",
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                params_dtype=ms.float16,
                quant_config=quant_config,
                prefix=f"{prefix}.project_in",
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = mint.nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
                dtype=ms.float16,
            )
        else:
            self.final_layer_norm = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OPTDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

    def get_input_embeddings(self, input_ids: ms.Tensor) -> ms.Tensor:
        return self.embed_tokens(input_ids)

    def construct(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        key_caches: List[ms.Tensor],
        value_caches: List[ms.Tensor],
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: ms.Tensor,
        batch_valid_length: ms.Tensor,
        q_seq_lens: ms.Tensor,
        block_tables: ms.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> Union[ms.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            pos_embeds = self.embed_positions(positions)
            if num_prefill_tokens > 0:
                pos_embeds = pos_embeds.expand_dims(0)
            else:
                pos_embeds = pos_embeds.expand_dims(1)
            if self.project_in is not None:
                inputs_embeds, _ = self.project_in(inputs_embeds)
            hidden_states = mint.add(inputs_embeds, pos_embeds)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states,
                key_caches[i - self.start_layer],
                value_caches[i - self.start_layer],
                num_prefill_tokens,
                num_decode_tokens,
                slot_mapping,
                batch_valid_length,
                q_seq_lens,
                block_tables,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Cell):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.decoder = OPTDecoder(
            config, cache_config, quant_config, prefix=f"{prefix}.decoder"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: ms.Tensor) -> ms.Tensor:
        return self.decoder.get_input_embeddings(input_ids)

    @ms.jit(jit_level="O0", infer_boost="on")
    def construct(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        key_caches: List[ms.Tensor],
        value_caches: List[ms.Tensor],
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: ms.Tensor,
        batch_valid_length: ms.Tensor,
        q_seq_lens: ms.Tensor,
        block_tables: ms.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> Union[ms.Tensor, IntermediateTensors]:
        return self.decoder(
            input_ids,
            positions,
            key_caches,
            value_caches,
            num_prefill_tokens,
            num_decode_tokens,
            slot_mapping,
            batch_valid_length,
            q_seq_lens,
            block_tables,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


class OPTForCausalLM(MsModelBase, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.word_embed_proj_dim, params_dtype=ms.float16
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.set_model_inputs()
        self.kv_caches = [
            Fake_Attention(dtype=ms.float16) for i in range(config.num_hidden_layers)
        ]
        compilation_config = vllm_config.compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(config.num_hidden_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

    def get_input_embeddings(self, input_ids: ms.Tensor) -> ms.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        kv_caches: List[ms.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Union[ms.Tensor, IntermediateTensors]:
        key_cache, value_cache = self.get_kvcache()
        if attn_metadata.num_prefill_tokens > 0:
            input_ids = input_ids.expand_dims(0)
        if attn_metadata.num_decode_tokens > 0:
            input_ids = input_ids.expand_dims(1)
        num_prefill_tokens = ms.mutable(attn_metadata.num_prefill_tokens)
        num_decode_tokens = ms.mutable(attn_metadata.num_decode_tokens)
        slot_mapping = attn_metadata.slot_mapping
        batch_valid_length = ms.Tensor.from_numpy(
            np.array(attn_metadata.seq_lens, dtype=np.int32)
        )
        q_seq_lens = ms.Tensor.from_numpy(
            np.array(attn_metadata.query_lens, dtype=np.int32)
        )
        block_tables = attn_metadata.block_tables

        model_output = self.model(
            input_ids,
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
            inputs_embeds=inputs_embeds,
        )
        if attn_metadata.num_prefill_tokens > 0:
            model_output = model_output.squeeze(0)
        if attn_metadata.num_decode_tokens > 0:
            model_output = model_output.squeeze(1)
        return model_output

    def compute_logits(
        self,
        hidden_states: ms.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[ms.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: ms.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, ms.Tensor]]) -> Set[str]:
        params_dict = self.get_params_dict()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                continue

            if name.startswith("decoder."):
                name = "model." + name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # no matter it is tied or not, we assume the weight is loaded.
        loaded_params.add("lm_head.weight")
        return loaded_params
