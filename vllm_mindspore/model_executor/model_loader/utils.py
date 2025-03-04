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

from typing import Tuple, Type

from torch import nn

from vllm.config import ModelConfig

from vllm_mindspore.model_executor.models.registry import MindSporeModelRegistry


def get_ms_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    model_cls, arch = MindSporeModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        raise RecursionError("MindSpore unsupport embed model task now!")
    elif model_config.task == "classify":
        raise RecursionError("MindSpore unsupport classify model task now!")
    elif model_config.task == "reward":
        raise RecursionError("MindSpore unsupport reward model task now!")

    return model_cls, arch
