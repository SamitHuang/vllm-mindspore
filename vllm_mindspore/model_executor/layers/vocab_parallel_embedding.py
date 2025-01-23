#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import mindspore as ms
from mindspore import Tensor
import mindspore.mint.nn.functional as F
from mindspore import mint
from mindspore import Parameter
import numpy as np

# 需要有一个基于mindspore的并行基类， 可以查询rank ，world_size 的信息。
from vllm_mindspore.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm_mindspore.model_executor.utils import set_weight_attrs

DEFAULT_VOCAB_PADDING_SIZE = 64

# TODO(tronzhang): Most same as vllm's one, check latter...


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            np.zeros(
                (sum(output_partition_sizes), input_size_per_partition),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        # layer.register_parameter("weight", weight)
        layer.insert_param_to_cell("weight", weight)

        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer, input_):
        return F.embedding(input_, layer.weight)


def get_masked_input_and_mask(
    input_: Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> Tuple[Tensor, Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    # org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ <
    #                                                       org_vocab_end_index)
    org_vocab_mask = mint.logical_and(
        input_ >= org_vocab_start_index, input_ < org_vocab_end_index
    )
    # added_vocab_mask = (input_ >= added_vocab_start_index) & (
    #     input_ < added_vocab_end_index)
    added_vocab_mask = mint.logical_and(
        input_ >= added_vocab_start_index, input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = mint.logical_or(org_vocab_mask, added_vocab_mask)
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


class VocabParallelEmbedding(ms.nn.Cell):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype=None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        # Keep the input dimensions.
        tp_rank = get_tensor_model_parallel_rank()  #  获取tp并行的rank
        self.tp_size = get_tensor_model_parallel_world_size()  #  获取tp并行的world_size
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )

        self.embedding_dim = embedding_dim

        linear_method = None
        if quant_config is not None:
            linear_method = quant_config.get_quant_method(self, prefix=prefix)
        if linear_method is None:
            linear_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self.__class__) is VocabParallelEmbedding
        linear_method_implements_embedding = method_has_implemented_embedding(
            type(linear_method)
        )
        if is_embedding_layer and not linear_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(linear_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.linear_method: QuantizeMethodBase = linear_method

        if params_dtype is None:
            # params_dtype = torch.get_default_dtype()
            # TODO:
            import numpy as np

            params_dtype = np.float16
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.linear_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def construct(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.linear_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            # output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)  # 使用并行接口
        return output

    def weight_loader(self, param, loaded_weight):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        # If the parameter is a gguf weight, then load it directly.
        # if getattr(param, "is_gguf_weight_type", None):
        #     param.data.copy_(loaded_weight)
        #     param.weight_type = loaded_weight.item()
        #     return
        # elif isinstance(param, UninitializedParameter):
        #     shape = list(loaded_weight.shape)
        #     if output_dim is not None:
        #         shape[output_dim] = shape[output_dim] // self.tp_size
        #     param.materialize(tuple(shape), dtype=loaded_weight.dtype)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            assert param.data.shape == loaded_weight.shape
            # param.data.copy_(loaded_weight)
            param.set_data(loaded_weight)
            return

        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        # if packed_dim is not None and packed_dim == output_dim:
        #     packed_factor = param.packed_factor if isinstance(
        #         param, BasevLLMParameter) else param.pack_factor
        #     assert loaded_weight.shape[output_dim] == (self.org_vocab_size //
        #                                                param.packed_factor)
        #     start_idx = start_idx // packed_factor
        #     shard_size = shard_size // packed_factor
        # else:
        #     assert loaded_weight.shape[output_dim] == self.org_vocab_size
        assert loaded_weight.shape[output_dim] == self.org_vocab_size

        # Copy the data.
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # if current_platform.is_hpu():
        # TODO: to support hpu
        if False:
            # FIXME(kzawora): Weight copy with slicing bugs out on Gaudi here,
            # so we're using a workaround. Remove this when fixed in
            # HPU PT bridge.
            # padded_weight = torch.cat([
            #     loaded_weight,
            #     torch.zeros(param.shape[0] - loaded_weight.shape[0],
            #                 *loaded_weight.shape[1:])
            # ])
            # param.data.copy_(padded_weight)
            ...
        else:
            # param[:loaded_weight.shape[0]].data.copy_(loaded_weight)
            # param[loaded_weight.shape[0]:].data.fill_(0)
            param[: loaded_weight.shape[0]] = loaded_weight
            param[loaded_weight.shape[0] :] = 0


class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype=None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(
                mint.zeros(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            # self.register_parameter("bias", None)
            self.bias = None

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            # self.weight = embed_tokens.weight
            self.weight.set_data(embed_tokens.weight)
            return self

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
