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

import contextlib
import gc
import logging
import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import numpy as np

import torch

if TYPE_CHECKING:
    from torch.library import Library
else:
    Library = None

from vllm.utils import T, TORCH_DTYPE_TO_NUMPY_DTYPE, make_ndarray_with_pad

import mindspore as ms
from mindspore.common.initializer import Zero
from mindspore import dtype as mstype
from mindspore.common.api import _pynative_executor

from .scripts import env_setup

MsKVCache = Tuple[ms.Tensor, ms.Tensor]

logger = logging.getLogger(__name__)


STR_DTYPE_TO_MS_DTYPE = {
    "half": ms.float16,
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "float": ms.float32,
    "fp8": ms.uint8,
    "fp8_e4m3": ms.uint8,
    "fp8_e5m2": ms.uint8,
}


def get_valid_dtype(dtype):
    if isinstance(dtype, str):
        dtype = STR_DTYPE_TO_MS_DTYPE[dtype]
    return dtype


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
): ...


@contextlib.contextmanager
def memory_profiling(
        baseline_snapshot: "MemorySnapshot",
        weights_memory: int) -> "Generator[MemoryProfilingResult, None, None]":
    """Memory profiling context manager.
    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the argument `weights_memory`.

    The increase of `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance until after profiling to get (c.).
    """ # noqa
    from vllm.utils import MemoryProfilingResult

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_create = baseline_snapshot
    # the part of memory used for holding the model weights
    result.weights_memory = weights_memory

    result.before_profile.measure()

    before_torch_memory_in_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    yield result

    gc.collect()
    torch.cuda.empty_cache()

    result.after_profile.measure()

    after_torch_memory_in_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = after_torch_memory_in_bytes - before_torch_memory_in_bytes
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = result.non_torch_increase + result.torch_peak_increase + result.weights_memory  # noqa


def _create_empty_tensor(ms_type):
    init_func = Zero()
    init_func.__enable_zero_dim__ = True
    init_tensor = ms.Tensor(shape=(0,), dtype=ms_type, init=init_func)
    init_tensor.init_data()

    return init_tensor


def make_tensor_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    pin_memory = False

    if padded_x.size == 0:
        tensor = _create_empty_tensor(dtype)
    else:
        tensor = torch.from_numpy(padded_x)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    if not data:
        t = _create_empty_tensor(dtype)
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="CPU")
    return t


STR_DTYPE_TO_TENSOR_DTYPE = {
    "half": torch.half,
    "float16": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

STR_DTYPE_TO_MS_DTYPE = {
    "half": mstype.float16,
    "float16": mstype.float16,
    "bfloat16": mstype.bfloat16,
    "float": mstype.float32,
    "fp8": mstype.uint8,
    "fp8_e4m3": mstype.uint8,
    "fp8_e5m2": mstype.uint8,
}


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    if isinstance(dtype, str):
        dtype = STR_DTYPE_TO_TENSOR_DTYPE[dtype]
    return torch.tensor([1], dtype=dtype).itemsize


def ascend_device_count_stateless() -> List[str]:
    visible_device_str = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
    if visible_device_str:
        try:
            res = visible_device_str.split(",")
        except Exception as e:
            logger.error('Cannot parse "ASCEND_RT_VISIBLE_DEVICES" for: %s!' % str(e))
            raise ValueError(
                'Error argument(%s) of environ "ASCEND_RT_VISIBLE_DEVICES"!'
                % visible_device_str
            )

        return len(res)

    import re
    import subprocess

    output = subprocess.check_output(["npu-smi", "info"], encoding="utf-8")
    res = re.findall(
        r"\|\s+\d+\s+\w+\s+\|\s+(\w+)\s+\|\s+(?:[0-9\.]+|-)\s+[0-9\.]+\s+\d+\s+\/\s+\d+\s+\|",
        output,
    )

    avl_devices = []
    for i, stat in enumerate(res):
        if stat != "OK":
            logger.warning("Device %d is not ok, status is %s!" % (i, stat))
        else:
            avl_devices.append(str(i))
    visible_device_str = ",".join(avl_devices)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_device_str
    logger.info('Set environ "ASCEND_RT_VISIBLE_DEVICES" as %s' % visible_device_str)

    return len(avl_devices)


def ascend_is_initialized():
    # Just return true for check.
    return True


def is_mindformers_model_backend():
    return (
        os.getenv("vLLM_MODEL_BACKEND")
        and os.environ["vLLM_MODEL_BACKEND"] == "MindFormers"
    )


def check_ready():
    import vllm.envs as envs
    from mindspore import set_context

    if envs.VLLM_USE_V1:
        raise NotImplementedError("vLLM-MindSpore does not support VLLM V1 now!")

    # Common environment variables of predict.
    set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    if is_mindformers_model_backend():
        logger.info("Run with Mindformers backend!")
        necessary_envs = ("vLLM_MODEL_MEMORY_USE_GB", "MINDFORMERS_MODEL_CONFIG")
        lost_envs = [env_item for env_item in necessary_envs if not os.getenv(env_item)]

        if lost_envs:
            raise RuntimeError(
                'For "MindFormers" model backend, environments %s should be set!'
                % str(lost_envs)
            )

        mindformers_default_env = {
            "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST": "FlashAttentionScore,PagedAttention",
            "MS_ALLOC_CONF": "enable_vmm:False",
        }
        env_setup(mindformers_default_env)
    else:
        env_setup({"MS_ALLOC_CONF": "enable_vmm:True", })
        logger.info("Run with native model backend!")


def calc_block_num(cache_config, model_config, parallel_config):
    from vllm.worker.cache_engine import CacheEngine

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_gpu_memory = int(os.environ["ASCEND_TOTAL_MEMORY_GB"]) if os.getenv("ASCEND_TOTAL_MEMORY_GB") else 64
    total_gpu_memory = total_gpu_memory * 1024 * 1024 * 1024
    memory_can_use = total_gpu_memory * cache_config.gpu_memory_utilization

    model_use_memory_b = int(os.getenv("vLLM_MODEL_MEMORY_USE_GB")) * 1024 * 1024 * 1024
    available_cache_memory = memory_can_use - model_use_memory_b
    cache_block_size = CacheEngine.get_cache_block_size(
        cache_config, model_config, parallel_config
    )
    num_gpu_blocks = int(available_cache_memory // cache_block_size)
    return num_gpu_blocks


def is_use_mla(model_config):
    if not is_mindformers_model_backend():
        return False

    return hasattr(model_config.hf_text_config, "model_type") and (
        model_config.hf_text_config.model_type in ("deepseek_v3",)
    )


def convert_np_to_ms_dtype(value):
    """convert_np_to_ms_dtype"""
    if value.dtype == np.int8:
        value_dtype = ms.int8
    elif value.dtype == np.int32:
        value_dtype = ms.int32
    elif value.dtype == np.int64:
        value_dtype = ms.int64
    elif value.dtype == np.float64:
        value_dtype = ms.float64
    elif value.dtype == np.float32:
        value_dtype = ms.float32
    else:
        value_dtype = ms.bfloat16
    return value_dtype