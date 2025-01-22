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

import os
import sys
import warnings

msadapter_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "msadapter/mindtorch")
)
sys.path.insert(0, msadapter_path)

from .version import __version__

if "vllm" in sys.modules:
    # Check models variable in sub process, cannot raise here.
    warnings.warn(
        "vllm import before vllm_mindspore, vllm_mindspore cannot worker right!"
    )

from vllm_mindspore.platforms.ascend import AscendPlatform

ascend_platform = AscendPlatform()

import vllm.config

vllm.config.current_platform = ascend_platform
import vllm.platforms

vllm.platforms.current_platform = ascend_platform
import vllm.utils

vllm.utils.current_platform = ascend_platform

from vllm_mindspore.utils import (
    direct_register_custom_op,
    memory_profiling,
    make_tensor_with_pad,
    async_tensor_h2d,
    get_dtype_size,
)

vllm.utils.direct_register_custom_op = direct_register_custom_op
vllm.utils.memory_profiling = memory_profiling
vllm.utils.make_tensor_with_pad = make_tensor_with_pad
vllm.utils.async_tensor_h2d = async_tensor_h2d
vllm.utils.get_dtype_size = get_dtype_size

from vllm_mindspore.model_executor.models.registry import (
    MindSporeModelRegistry,
    _run_in_subprocess,
)
import vllm.model_executor

vllm.model_executor.models.ModelRegistry = MindSporeModelRegistry
vllm.config.ModelRegistry = MindSporeModelRegistry

from vllm_mindspore.model_executor.model_loader.utils import get_ms_model_architecture

vllm.model_executor.model_loader.get_model_architecture = get_ms_model_architecture
vllm.model_executor.model_loader.utils.get_model_architecture = (
    get_ms_model_architecture
)
vllm.model_executor.model_loader.loader.get_model_architecture = (
    get_ms_model_architecture
)
vllm.model_executor.models.registry._run_in_subprocess = _run_in_subprocess

from vllm_mindspore.model_executor.model_loader import get_ms_model_loader, get_ms_model

vllm.model_executor.model_loader.get_model_loader = get_ms_model_loader
vllm.model_executor.model_loader.get_model = get_ms_model

from vllm_mindspore.model_executor.sampling_metadata import (
    SequenceGroupToSample,
    SamplingMetadataCache,
    SamplingMetadata,
)

vllm.model_executor.SamplingMetadataCache = SamplingMetadataCache
vllm.model_executor.SamplingMetadata = SamplingMetadata
vllm.model_executor.sampling_metadata.SequenceGroupToSample = SequenceGroupToSample
vllm.model_executor.sampling_metadata.SamplingMetadataCache = SamplingMetadataCache
vllm.model_executor.sampling_metadata.SamplingMetadata = SamplingMetadata

from vllm_mindspore.attention.selector import get_ms_attn_backend
import vllm.attention

vllm.attention.get_attn_backend = get_ms_attn_backend

from vllm_mindspore.worker.cache_engine import (
    ms_allocate_kv_cache,
    ms_swap_in,
    ms_swap_out,
)
import vllm.worker.cache_engine

vllm.worker.cache_engine.CacheEngine._allocate_kv_cache = ms_allocate_kv_cache
vllm.worker.cache_engine.CacheEngine.swap_in = ms_swap_in
vllm.worker.cache_engine.CacheEngine.swap_out = ms_swap_out

from vllm_mindspore.distributed.parallel_state import (
    initialize_model_parallel,
    init_distributed_environment,
    ensure_kv_transfer_initialized,
    model_parallel_is_initialized,
    ensure_model_parallel_initialized,
)


class PP:
    def __init__(self):
        self.is_first_rank = True
        self.is_last_rank = True


def get_pp_group():
    return PP()


import vllm.distributed.parallel_state

vllm.distributed.parallel_state.get_pp_group = get_pp_group
vllm.distributed.parallel_state.initialize_model_parallel = initialize_model_parallel
vllm.distributed.parallel_state.init_distributed_environment = (
    init_distributed_environment
)
vllm.distributed.parallel_state.ensure_kv_transfer_initialized = (
    ensure_kv_transfer_initialized
)
vllm.distributed.parallel_state.model_parallel_is_initialized = (
    model_parallel_is_initialized
)
vllm.distributed.parallel_state.ensure_model_parallel_initialized = (
    ensure_model_parallel_initialized
)

vllm.distributed.get_pp_group = get_pp_group
vllm.distributed.init_distributed_environment = init_distributed_environment
vllm.distributed.ensure_kv_transfer_initialized = ensure_kv_transfer_initialized
vllm.distributed.model_parallel_is_initialized = model_parallel_is_initialized
vllm.distributed.ensure_model_parallel_initialized = ensure_model_parallel_initialized

from vllm_mindspore.worker.worker import (
    _warm_up_model,
    determine_num_available_blocks,
    prepare_worker_input,
)
from vllm.worker.worker import Worker

Worker._warm_up_model = _warm_up_model
Worker.determine_num_available_blocks = determine_num_available_blocks
Worker.prepare_worker_input = prepare_worker_input
vllm.worker.worker_base.get_pp_group = get_pp_group

from vllm_mindspore.worker.model_runner import _get_cuda_graph_pad_size

vllm.worker.model_runner.ModelInputForGPUBuilder._get_cuda_graph_pad_size = (
    _get_cuda_graph_pad_size
)
