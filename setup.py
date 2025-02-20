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
"""setup package."""

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


if not sys.platform.startswith("linux"):
    logger.warning(
        "vllm_mindspore only supports Linux platform."
        "Building on %s, "
        "so vllm_mindspore may not be able to run correctly",
        sys.platform,
    )


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


def prepare_submodules() -> None:
    def _run_cmd(args: str, check: bool = True) -> None:
        cmds = args.split(" ")
        returned = subprocess.run(cmds, stderr=subprocess.STDOUT)
        if check:
            returned.check_returncode()
        elif returned.returncode != 0:
            logger.warning("Run %s error, please check!" % args)

    old_dir = os.getcwd()
    os.chdir(ROOT_DIR)

    msadapter_path = Path() / "vllm_mindspore" / "msadapter"
    if not any(msadapter_path.iterdir()):
        # Fetch source codes.
        _run_cmd("git submodule update --init vllm_mindspore/msadapter", check=False)
        os.chdir(get_path("vllm_mindspore", "msadapter"))
        _run_cmd("git reset --hard HEAD")

        patch_dir = Path(ROOT_DIR) / "patch" / "msadapter"
        for p in patch_dir.glob("*.patch"):
            _run_cmd("git apply {}".format(p))

        # Add __init__.py for packing.
        _run_cmd("touch __init__.py")
        _run_cmd("touch mindtorch/__init__.py")

    os.chdir(old_dir)


prepare_submodules()

setup(
    name="vllm-mindspore",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    author="MindSpore Team",
    license="Apache 2.0",
    description=(
        "A high-throughput and memory-efficient inference and "
        "serving engine for LLMs"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://gitee.com/mindspore/vllm_mindspore",
    project_urls={
        "Homepage": "https://gitee.com/mindspore/vllm_mindspore",
        "Documentation": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    # entry_points={
    #     "console_scripts": [
    #         "vllm_mindspore=vllm_mindspore.scripts:main",
    #     ],
    # },
)
