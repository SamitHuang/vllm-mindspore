from typing import cast
import torch
from vllm.multimodal.inputs import BatchedTensorInputs
from vllm.utils import JSONTree, json_map_leaves


def as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device: torch.types.Device,
) -> BatchedTensorInputs:
    json_inputs = cast(JSONTree[torch.Tensor], batched_inputs)

    json_mapped = json_map_leaves(
        lambda x: x,
        json_inputs,
    )

    return cast(BatchedTensorInputs, json_mapped)
