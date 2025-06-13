# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import cached_property
from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from transformers import Blip2QFormerConfig
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.models.blip2 import (
    _IMAGE_TOKEN_ID,
    Blip2DummyInputsBuilder,
    Blip2ImageEmbeddingInputs,
    Blip2ImageInputs,
    Blip2ImagePixelInputs,
    Blip2MultiModalProcessor,
    Blip2ProcessingInfo,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.multimodal import MULTIMODAL_REGISTRY, NestedTensors
from vllm.sequence import IntermediateTensors
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm_mindspore.model_executor.layers.activation import get_act_fn
from vllm_mindspore.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm_mindspore.model_executor.models.interfaces import SupportsMultiModal
from vllm_mindspore.model_executor.models.model_base import MsModelBase
from vllm_mindspore.model_executor.models.utils import (
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata

from .blip import BlipVisionModel
from .opt import OPTModel


def apply_chunking_to_forward(
    forward_fn: Callable[..., ms.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors: ms.Tensor,
) -> ms.Tensor:
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(
            input_tensor.chunk(num_chunks, dim=chunk_dim)
            for input_tensor in input_tensors
        )
        # apply forward fn to every tuple
        output_chunks = tuple(
            forward_fn(*input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks)
        )
        # concatenate output at same dimension
        return mint.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class Blip2QFormerMultiHeadAttention(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        is_cross_attention: bool = False,
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

        self.query = mint.nn.Linear(
            config.hidden_size, self.all_head_size, dtype=ms.bfloat16
        )
        if is_cross_attention:
            kv_hidden_size = config.encoder_hidden_size
        else:
            kv_hidden_size = config.hidden_size
        self.key = mint.nn.Linear(kv_hidden_size, self.all_head_size, dtype=ms.bfloat16)
        self.value = mint.nn.Linear(
            kv_hidden_size, self.all_head_size, dtype=ms.bfloat16
        )

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

        attention_scores = mint.matmul(query_layer, key_layer.transpose(-1, -2))
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

    def __init__(self, config: Blip2QFormerConfig) -> None:
        super().__init__()

        self.dense = mint.nn.Linear(
            config.hidden_size, config.hidden_size, dtype=ms.bfloat16
        )
        self.LayerNorm = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=ms.bfloat16
        )
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Blip2QFormerAttention(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.attention = Blip2QFormerMultiHeadAttention(
            config,
            quant_config=quant_config,
            cache_config=cache_config,
            is_cross_attention=is_cross_attention,
        )

        self.output = Blip2QFormerSelfOutput(config)

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

    def __init__(self, config: Blip2QFormerConfig) -> None:
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size, config.intermediate_size, dtype=ms.bfloat16
        )
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Blip2QFormerOutput(nn.Cell):

    def __init__(self, config: Blip2QFormerConfig) -> None:
        super().__init__()

        self.dense = mint.nn.Linear(
            config.intermediate_size, config.hidden_size, dtype=ms.bfloat16
        )
        self.LayerNorm = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=ms.bfloat16
        )
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(
        self,
        hidden_states: ms.Tensor,
        input_tensor: ms.Tensor,
    ) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Blip2QFormerLayer(nn.Cell):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(
            config, quant_config=quant_config, cache_config=cache_config
        )

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(
                config,
                quant_config=quant_config,
                cache_config=cache_config,
                is_cross_attention=True,
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)

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
    ) -> None:
        super().__init__()

        self.config = config

        self.layernorm = mint.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=ms.bfloat16
        )
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(
            config, quant_config=quant_config, cache_config=cache_config
        )

    def construct(
        self,
        query_embeds: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
    ) -> ms.Tensor:
        query_length = query_embeds.shape[1]

        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)

        sequence_output = self.encoder(
            embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            query_length=query_length,
        )

        return sequence_output


@MULTIMODAL_REGISTRY.register_processor(
    Blip2MultiModalProcessor,
    info=Blip2ProcessingInfo,
    dummy_inputs=Blip2DummyInputsBuilder,
)
class Blip2ForConditionalGeneration(MsModelBase, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_model = BlipVisionModel(config.vision_config, quant_config)

        self.query_tokens = ms.Parameter(
            mint.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )

        self.qformer = Blip2QFormerModel(
            config.qformer_config, cache_config=cache_config, quant_config=quant_config
        )

        self.language_projection = mint.nn.Linear(
            config.qformer_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
            dtype=ms.bfloat16,
        )

        self.language_model = OPTModel(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
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
            if not isinstance(pixel_values, ms.Tensor):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            # Remove the N dimension until multiple images are supported.
            pixel_values = pixel_values.squeeze(1)

            return Blip2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, ms.Tensor):
                raise ValueError(
                    "Incorrect type of image embeddings. "
                    f"Got type: {type(image_embeds)}"
                )

            # Remove the N dimension until multiple images are supported.
            image_embeds = image_embeds.squeeze(1)

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

        query_tokens = self.query_tokens.expand(image_features.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
        )

        return self.language_projection(query_output)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: ms.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> ms.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, _IMAGE_TOKEN_ID
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: ms.Tensor,
        positions: ms.Tensor,
        kv_caches: List[ms.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:
        """Run forward pass for BLIP-2.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"Question: What's the content of the image? Answer:"`.

        Tokenizer outputs:
        `[2, 45641, 35, 653, 18, 5, 1383, 9, 5, 2274, 116, 31652, 35]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        dummy tokens (denoted as `50265`), resulting in:
        `[50265, ..., 50265, 2, 45641, 35, ..., 31652, 35]`.

        We insert 32 tokens since it corresponds to the number of query
        embeddings outputted by the Q-Former and inputted to the language model.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each input image.

        See also:
            :class:`Blip2ImageInputs`
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
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
        for name, weight in weights:
            if "vision_model." in name:
                self.vision_model.load_weights([(name, weight)], params_dict)
            else:
                raise NotImplementedError()
