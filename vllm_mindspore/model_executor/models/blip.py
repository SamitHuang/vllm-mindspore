# SPDX-License-Identifier: Apache-2.0
"""Minimal implementation of BlipVisionModel intended to be only used
within a vision language model."""
import math
from typing import Dict, Iterable, Optional, Set, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from transformers import Blip2VisionConfig, BlipVisionConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm_mindspore.model_executor.layers.activation import get_act_fn
from vllm_mindspore.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)


def get_blip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size


def get_blip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_blip_patch_grid_length(
        image_size=image_size, patch_size=patch_size
    )
    return grid_length * grid_length


class BlipVisionEmbeddings(nn.Cell):

    def __init__(
        self,
        config: Union[BlipVisionConfig, Blip2VisionConfig],
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = ms.Parameter(
            mint.randn(1, 1, self.embed_dim, dtype=dtype)
        )

        self.patch_embedding = mint.nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            dtype=dtype,
        )

        self.num_patches = get_blip_num_patches(
            image_size=self.image_size, patch_size=self.patch_size
        )
        self.num_positions = self.num_patches + 1

        self.position_embedding = ms.Parameter(
            mint.randn(1, self.num_positions, self.embed_dim, dtype=dtype)
        )

    def construct(self, pixel_values: ms.Tensor) -> ms.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # shape = [*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = mint.transpose(patch_embeds.flatten(2), 1, 2)

        class_embeds = mint.broadcast_to(self.class_embedding, (batch_size, 1, -1))
        embeddings = mint.cat([class_embeds, patch_embeds], dim=1)

        position_embeds = self.position_embedding.to(target_dtype)
        embeddings = mint.add(embeddings, position_embeds[:, : embeddings.shape[1], :])

        return embeddings


class BlipAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Union[BlipVisionConfig, Blip2VisionConfig],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            params_dtype=dtype,
        )
        self.projection = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.projection",
            params_dtype=dtype,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)
        self.padding_num = 16 - self.head_dim % 16

    def attn(self, query: ms.Tensor, key: ms.Tensor, value: ms.Tensor):
        b, q_len, _ = query.shape
        _, k_len, _ = key.shape

        query = query.view(b, q_len, self.num_heads // self.tp_size, self.head_dim)
        key = key.view(b, k_len, self.num_heads // self.tp_size, self.head_dim)
        value = value.view(b, k_len, self.num_heads // self.tp_size, self.head_dim)

        if self.padding_num > 0:
            query = F.pad(query, (0, self.padding_num))
            key = F.pad(key, (0, self.padding_num))
            value = F.pad(value, (0, self.padding_num))

        # under jit, it does not support BSND, converted back to BSH
        query = query.view(b, q_len, -1)
        key = key.view(b, k_len, -1)
        value = value.view(b, k_len, -1)

        out = ops.flash_attention_score(
            query,
            key,
            value,
            self.num_heads // self.tp_size,
            scalar_value=1 / math.sqrt(self.head_dim),
            input_layout="BSH",
        )

        out = out.view(b, q_len, self.num_heads // self.tp_size, -1)

        if self.padding_num > 0:
            out = out[..., : self.head_dim]

        out = out.view(b, q_len, -1)
        return out

    def construct(
        self,
        hidden_states: ms.Tensor,
    ) -> Tuple[ms.Tensor, None]:
        """Input shape: Batch x Time x Channel"""

        qkv_states, _ = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.projection(out)

        return attn_output, None


class BlipMLP(nn.Cell):

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.config = config

        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            params_dtype=dtype,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            params_dtype=dtype,
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class BlipEncoderLayer(nn.Cell):

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        # fallback to sdpa attention if tp unavailable
        self.self_attn = BlipAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn", dtype=dtype
        )
        self.layer_norm1 = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.mlp = BlipMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp", dtype=dtype
        )
        self.layer_norm2 = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = mint.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = mint.add(residual, hidden_states)

        return hidden_states


class BlipEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`BlipEncoderLayer`].

    Args:
        config: BlipConfig
    """

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.CellList(
            [
                BlipEncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    dtype=dtype,
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def construct(self, inputs_embeds: ms.Tensor) -> ms.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class BlipVisionModel(nn.Cell, SupportsQuant):
    config_class = BlipVisionConfig
    main_input_name = "pixel_values"
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.config = config

        self.embeddings = BlipVisionEmbeddings(config, dtype=dtype)
        self.encoder = BlipEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
            dtype=dtype,
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = mint.nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
            )
        else:
            self.post_layernorm = None

    def construct(self, pixel_values: ms.Tensor) -> ms.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(inputs_embeds=hidden_states)

        if self.post_layernorm is None:
            return hidden_states

        return self.post_layernorm(hidden_states)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, ms.Tensor]],
        params_dict: Dict[str, ms.Parameter],
    ) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        loaded_params: Set[str] = set()
        layer_count = len(self.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in BlipVisionModel
            if name.startswith("post_layernorm") and self.post_layernorm is None:
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("encoder.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
