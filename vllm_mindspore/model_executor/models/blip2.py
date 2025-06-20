# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Iterable, Mapping, Optional, Set, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindone.transformers.mindspore_utils import apply_chunking_to_forward
from transformers import BatchFeature, Blip2QFormerConfig
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.blip2 import (
    _IMAGE_TOKEN_ID,
    Blip2DummyInputsBuilder,
    Blip2ImageEmbeddingInputs,
    Blip2ImageInputs,
    Blip2ImagePixelInputs,
    Blip2MultiModalProcessor,
    Blip2ProcessingInfo,
)
from vllm.model_executor.models.interfaces import SupportsPP, SupportsQuant
from vllm.model_executor.models.utils import flatten_bn
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm_mindspore.model_executor.layers.activation import get_act_fn
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm_mindspore.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm_mindspore.model_executor.models.attention_mask import LowerTriangularMask
from vllm_mindspore.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm_mindspore.model_executor.models.model_base import (
    AttentionWrapper,
    NativeModel,
)
from vllm_mindspore.model_executor.models.utils import (
    maybe_prefix,
    merge_multimodal_embeddings,
)

from .blip import BlipVisionModel
from .opt import OPTForCausalLM


class Blip2MultiModalProcessor_(Blip2MultiModalProcessor):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            # HF processor always adds placeholders even when there's no image
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="np")

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )


class Blip2QFormerMultiHeadAttention(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        is_cross_attention: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of "
                f"the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size, dtype=dtype)
        if is_cross_attention:
            kv_hidden_size = config.encoder_hidden_size
        else:
            kv_hidden_size = config.hidden_size
        self.key = mint.nn.Linear(kv_hidden_size, self.all_head_size, dtype=dtype)
        self.value = mint.nn.Linear(kv_hidden_size, self.all_head_size, dtype=dtype)

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type != "absolute":
            raise NotImplementedError(
                "Unsupported position_embedding_type: "
                f"{self.position_embedding_type}"
            )

        self.dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: ms.Tensor) -> ms.Tensor:
        x = x.view(*x.shape[:-1], self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = mint.matmul(query_layer, mint.transpose(key_layer, -1, -2))
        attention_probs = mint.softmax(attention_scores * self.scaling, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        context_layer = mint.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            *context_layer.shape[:-2], self.all_head_size
        )

        return context_layer


class Blip2QFormerSelfOutput(nn.Cell):

    def __init__(self, config: Blip2QFormerConfig, dtype: ms.Type = ms.float32) -> None:
        super().__init__()

        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.LayerNorm = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(mint.add(hidden_states, input_tensor))
        return hidden_states


class Blip2QFormerAttention(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        is_cross_attention: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.attention = Blip2QFormerMultiHeadAttention(
            config,
            quant_config=quant_config,
            cache_config=cache_config,
            is_cross_attention=is_cross_attention,
            dtype=dtype,
        )

        self.output = Blip2QFormerSelfOutput(config, dtype=dtype)

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        self_output = self.attention(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class Blip2QFormerIntermediate(nn.Cell):

    def __init__(self, config: Blip2QFormerConfig, dtype: ms.Type = ms.float32) -> None:
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size, config.intermediate_size, dtype=dtype
        )
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Blip2QFormerOutput(nn.Cell):

    def __init__(self, config: Blip2QFormerConfig, dtype: ms.Type = ms.float32) -> None:
        super().__init__()

        self.dense = mint.nn.Linear(
            config.intermediate_size, config.hidden_size, dtype=dtype
        )
        self.LayerNorm = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(
        self,
        hidden_states: ms.Tensor,
        input_tensor: ms.Tensor,
    ) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(mint.add(hidden_states, input_tensor))
        return hidden_states


class Blip2QFormerLayer(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        layer_idx: int,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(
            config, quant_config=quant_config, cache_config=cache_config, dtype=dtype
        )

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(
                config,
                quant_config=quant_config,
                cache_config=cache_config,
                is_cross_attention=True,
                dtype=dtype,
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate_query = Blip2QFormerIntermediate(config, dtype=dtype)
        self.output_query = Blip2QFormerOutput(config, dtype=dtype)

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
    ) -> ms.Tensor:
        attention_output = self.attention(hidden_states)

        if self.has_cross_attention:
            attention_output = self.crossattention(
                attention_output,
                encoder_hidden_states=encoder_hidden_states,
            )

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk_query,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

        return layer_output

    def feed_forward_chunk_query(self, attention_output: ms.Tensor) -> ms.Tensor:
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class Blip2QFormerEncoder(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.config = config

        self.layer = nn.CellList(
            [
                Blip2QFormerLayer(
                    config,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    layer_idx=layer_idx,
                    dtype=dtype,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def construct(
        self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor
    ) -> ms.Tensor:
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]

            hidden_states = layer_module(
                hidden_states, encoder_hidden_states=encoder_hidden_states
            )

        return hidden_states


class Blip2QFormerModel(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.config = config

        self.layernorm = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(
            config, quant_config=quant_config, cache_config=cache_config, dtype=dtype
        )

    def construct(
        self,
        query_embeds: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
    ) -> ms.Tensor:
        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)

        sequence_output = self.encoder(
            embedding_output, encoder_hidden_states=encoder_hidden_states
        )

        return sequence_output


@MULTIMODAL_REGISTRY.register_processor(
    Blip2MultiModalProcessor_,
    info=Blip2ProcessingInfo,
    dummy_inputs=Blip2DummyInputsBuilder,
)
class Blip2ForConditionalGeneration(
    NativeModel, SupportsMultiModal, SupportsPP, SupportsQuant
):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        dtype = self.model_config.dtype

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_model = BlipVisionModel(
            config.vision_config, quant_config, dtype=dtype
        )

        self.query_tokens = ms.Parameter(
            mint.zeros(
                (1, config.num_query_tokens, config.qformer_config.hidden_size),
                dtype=dtype,
            )
        )

        self.qformer = Blip2QFormerModel(
            config.qformer_config,
            cache_config=cache_config,
            quant_config=quant_config,
            dtype=dtype,
        )

        self.language_projection = mint.nn.Linear(
            config.qformer_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
            dtype=dtype,
        )

        self.language_model = OPTForCausalLM(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.common_preprocess(vllm_config, prefix)
        if self.is_graph_mode:
            self.vision_model = ms.jit(function=self.vision_model)
            self.vision_model.set_inputs(
                ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
            )

            self.qformer = ms.jit(function=self.qformer)
            self.qformer.set_inputs(
                ms.Tensor(shape=[None, None, None], dtype=self.model_config.dtype),
                ms.Tensor(shape=[None, None, None], dtype=self.model_config.dtype),
            )

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_pixel_values(self, data: ms.Tensor) -> ms.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}."
            )

        return data

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[Blip2ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (ms.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            pixel_values = flatten_bn(pixel_values, concat=True)

            return Blip2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (ms.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image embeddings. "
                    f"Got type: {type(image_embeds)}"
                )

            image_embeds = flatten_bn(image_embeds, concat=True)

            return Blip2ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self, vision_model: BlipVisionModel, pixel_values: ms.Tensor
    ) -> ms.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_model(pixel_values)

        return image_features

    def _process_image_pixels(self, inputs: Blip2ImagePixelInputs) -> ms.Tensor:
        assert self.vision_model is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_model, pixel_values)

    def _process_image_input(self, image_input: Blip2ImageInputs) -> ms.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None
        image_features = self._process_image_pixels(image_input)

        query_tokens = mint.broadcast_to(
            self.query_tokens, (image_features.shape[0], -1, -1)
        )
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
        )

        return self.language_projection(query_output)

    def get_language_model(self) -> nn.Cell:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> ms.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, _IMAGE_TOKEN_ID
            )
        return inputs_embeds

    def common_preprocess(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.set_modules(
            {
                "vision_model": self.vision_model,
                "qformer": self.qformer,
                "query_tokens": self.query_tokens,
                "language_projection": self.language_projection,
                "language_model": self.language_model,
            }
        )

        self.casual_mask = LowerTriangularMask(
            dtype=self.model_config.dtype, max_model_len=self.model_config.max_model_len
        )
        self.kv_caches = [
            AttentionWrapper()
            for _ in range(self.config.get_text_config().num_hidden_layers)
        ]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.config.get_text_config().num_hidden_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

    def exec_model(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        intermediate_tensors: IntermediateTensors = None,
        inputs_embeds: ms.Tensor = None,
        **kwargs,
    ):
        model_inputs, is_prefill = self.prepare_inputs(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        if self.prev_prefill != is_prefill and self.is_graph_mode:
            self.language_model.set_model_inputs(
                input_ids, positions, intermediate_tensors, inputs_embeds, is_prefill
            )
        self.prev_prefill = is_prefill

        # for dummy_attention_metadata
        if is_prefill and not self.set_flags:
            self.set_flags = True

        if self.run_model is None:
            self.run_model = (
                ms.jit(function=self.language_model.model)
                if self.is_graph_mode
                else self.language_model.model
            )
        model_output = self.run_model(
            input_ids=model_inputs["input_ids"],
            positions=model_inputs["position_ids"],
            key_caches=model_inputs["key_cache"],
            value_caches=model_inputs["value_cache"],
            is_prefill=is_prefill,
            slot_mapping=model_inputs["slot_mapping"],
            attn_mask=model_inputs["attention_mask"],
            batch_valid_length=model_inputs["batch_valid_length"],
            q_seq_lens=model_inputs["q_seq_lens"],
            block_tables=model_inputs["block_tables"],
            intermediate_tensors=model_inputs["intermediate_tensors"],
            inputs_embeds=model_inputs["inputs_embeds"],
        )

        return model_output

    def get_kvcache(self):
        key_cache = []
        value_cache = []
        forward_context = get_forward_context()
        for i in range(self.config.get_text_config().num_hidden_layers):
            k_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            v_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][1]
            key_cache.append(k_cache)
            value_cache.append(v_cache)
        return ms.mutable(key_cache), ms.mutable(value_cache)

    def forward(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.exec_model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: ms.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[ms.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: ms.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, ms.Tensor]]) -> Set[str]:
        params_dict = self.get_params_dict()
        dtype = self.model_config.dtype
        for name, weight in weights:
            if "vision_model." in name:
                self.vision_model.load_weights([(name, weight.to(dtype))], params_dict)
            elif "language_model." in name:
                self.language_model.load_weights(
                    [(name.replace("language_model.", ""), weight.to(dtype))]
                )
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight.to(dtype))
