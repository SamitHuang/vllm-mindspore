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

from typing import Dict, List, Optional
from collections import defaultdict

import vllm.envs as envs
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port)
from vllm.logger import init_logger
from vllm.config import ParallelConfig
from vllm.platforms import current_platform
from vllm.executor.ray_utils import RayWorkerWrapper, ray, available_resources_per_node
from vllm.executor.ray_gpu_executor import PlacementGroupSchedulingStrategy


logger = init_logger(__name__)


class MsRayWorkerWrapper(RayWorkerWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def ms_init_workers_ray(self, placement_group: "PlacementGroup",
                      **ray_remote_kwargs):
    if (self.parallel_config.tensor_parallel_size == 1
            and self.parallel_config.pipeline_parallel_size == 1):
        # For single GPU case, we use a ray worker with constrained memory.
        num_gpus = self.cache_config.gpu_memory_utilization
    else:
        # Otherwise, the ray workers are allocated with a full GPU.
        num_gpus = 1

    # The driver dummy worker does not actually use any resources.
    # It holds the resource for the driver worker.
    self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
    # The remaining workers are the actual ray actors.
    self.workers: List[RayWorkerWrapper] = []

    # Used in ray compiled DAG: indexed first by PP rank,
    # and then TP rank. In other words, the inner list is
    # the TP group of workers for a PP rank.
    self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

    if self.parallel_config.ray_workers_use_nsight:
        ray_remote_kwargs = self._configure_ray_workers_use_nsight(
            ray_remote_kwargs)

    logger.info("use_ray_spmd_worker: %s", self.use_ray_spmd_worker)

    # Create the workers.
    driver_ip = get_ip()
    workers = []
    for bundle_id, bundle in enumerate(placement_group.bundle_specs):
        if not bundle.get("NPU", 0):
            continue
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_id,
        )

        worker = ray.remote(
            num_cpus=0,
            num_gpus=0,
            resources={"NPU": 1},
            scheduling_strategy=scheduling_strategy,
            **ray_remote_kwargs,
        )(MsRayWorkerWrapper).remote(vllm_config=self.vllm_config)
        workers.append(worker)

    worker_ip_refs = [
        worker.get_node_ip.remote()  # type: ignore[attr-defined]
        for worker in workers
    ]
    worker_ips = ray.get(worker_ip_refs)

    if not self.use_ray_spmd_worker:
        for i in range(len(workers)):
            worker = workers[i]
            worker_ip = worker_ips[i]
            if self.driver_dummy_worker is None and worker_ip == driver_ip:
                # If the worker is on the same node as the driver, we use it
                # as the resource holder for the driver process.
                self.driver_dummy_worker = worker
                self.driver_worker = MsRayWorkerWrapper(
                    vllm_config=self.vllm_config)
                workers.pop(i)
                worker_ips.pop(i)
                self.workers = workers
                break
    else:
        self.workers = workers

    logger.debug("workers: %s", self.workers)
    logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)
    if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
        raise ValueError(
            "Ray does not allocate any GPUs on the driver node. Consider "
            "adjusting the Ray placement group or running the driver on a "
            "NPU node.")

    ip_counts: Dict[str, int] = {}
    for ip in worker_ips:
        ip_counts[ip] = ip_counts.get(ip, 0) + 1

    worker_to_ip = dict(zip(self.workers, worker_ips))

    def sort_by_driver_then_worker_ip(worker):
        """
        Sort the workers based on 3 properties:
        1. If the worker is on the same node as the driver (vllm engine),
            it should be placed first.
        2. Then, if the worker is on a node with fewer workers, it should
            be placed first.
        3. Finally, if the work is on a node with smaller IP address, it
            should be placed first.
        """
        ip = worker_to_ip[worker]
        return (ip != driver_ip, ip_counts[ip], ip)

    # After sorting, the workers on the same node will be
    # close to each other, and the workers on the driver
    # node will be placed first.
    self.workers = sorted(self.workers, key=sort_by_driver_then_worker_ip)

    # Get the set of GPU IDs used on each node.
    worker_node_and_gpu_ids = []
    for worker in [self.driver_dummy_worker] + self.workers:
        if worker is None:
            # driver_dummy_worker can be None when using ray spmd worker.
            continue
        worker_node_and_gpu_ids.append(
            ray.get(worker.get_node_and_gpu_ids.remote()) \
        ) # type: ignore

    node_workers = defaultdict(list)  # node id -> list of worker ranks
    node_gpus = defaultdict(list)  # node id -> list of gpu ids

    for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
        node_workers[node_id].append(i)
        # `gpu_ids` can be a list of strings or integers.
        # convert them to integers for consistency.
        # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
        # string sorting is not sufficient.
        # see https://github.com/vllm-project/vllm/issues/5590
        gpu_ids = [int(x) for x in gpu_ids]
        node_gpus[node_id].extend(gpu_ids)
    for node_id, gpu_ids in node_gpus.items():
        node_gpus[node_id] = sorted(gpu_ids)

    all_ips = set(worker_ips + [driver_ip])
    n_ips = len(all_ips)
    n_nodes = len(node_workers)

    if n_nodes != n_ips:
        raise RuntimeError(
            f"Every node should have a unique IP address. Got {n_nodes}"
            f" nodes with node ids {list(node_workers.keys())} and "
            f"{n_ips} unique IP addresses {all_ips}. Please check your"
            " network configuration. If you set `VLLM_HOST_IP`"
            " environment variable, make sure it is unique for"
            " each node.")

    # Set environment variables for the driver and workers.
    all_args_to_update_environment_variables = [({
        "CUDA_VISIBLE_DEVICES":
        ",".join(map(str, node_gpus[node_id])),
        "VLLM_TRACE_FUNCTION":
        str(envs.VLLM_TRACE_FUNCTION),
        **({
            "VLLM_ATTENTION_BACKEND": envs.VLLM_ATTENTION_BACKEND
        } if envs.VLLM_ATTENTION_BACKEND is not None else {})
    }, ) for (node_id, _) in worker_node_and_gpu_ids]

    self._env_vars_for_all_workers = (
        all_args_to_update_environment_variables)

    self._run_workers("update_environment_variables",
                      all_args=self._get_env_vars_to_be_updated())

    if len(node_gpus) == 1:
        # in single node case, we don't need to get the IP address.
        # the loopback address is sufficient
        # NOTE: a node may have several IP addresses, one for each
        # network interface. `get_ip()` might return any of them,
        # while they might not work for communication inside the node
        # if the network setup is complicated. Using the loopback address
        # solves this issue, as it always works for communication inside
        # the node.
        driver_ip = "127.0.0.1"
    distributed_init_method = get_distributed_init_method(
        driver_ip, get_open_port())

    # Initialize the actual workers inside worker wrapper.
    init_worker_all_kwargs = [
        self._get_worker_kwargs(
            local_rank=node_workers[node_id].index(rank),
            rank=rank,
            distributed_init_method=distributed_init_method,
        ) for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
    ]
    self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)

    self._run_workers("init_device")
    self._run_workers("load_model",
                      max_concurrent_workers=self.parallel_config.
                      max_parallel_loading_workers)

    if self.use_ray_spmd_worker:
        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(
                    self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank * self.parallel_config.tensor_parallel_size
                        ) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.workers[rank])

    # This is the list of workers that are rank 0 of each TP group EXCEPT
    # global rank 0. These are the workers that will broadcast to the
    # rest of the workers.
    self.tp_driver_workers: List[RayWorkerWrapper] = []
    # This is the list of workers that are not drivers and not the first
    # worker in a TP group. These are the workers that will be
    # broadcasted to.
    self.non_driver_workers: List[RayWorkerWrapper] = []

    # Enforce rank order for correct rank to return final output.
    for index, worker in enumerate(self.workers):
        # The driver worker is rank 0 and not in self.workers.
        rank = index + 1
        if rank % self.parallel_config.tensor_parallel_size == 0:
            self.tp_driver_workers.append(worker)
        else:
            self.non_driver_workers.append(worker)


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    from vllm.executor.ray_utils import assert_ray_available, _wait_until_pg_ready, _verify_bundles

    assert_ray_available()

    # Connect to a ray cluster.
    if current_platform.is_rocm() or current_platform.is_xpu():
        # Try to connect existing ray instance and create a new one if not found
        try:
            ray.init("auto", ignore_reinit_error=True)
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources.")
            ray.init(address=ray_address,
                     ignore_reinit_error=True,
                     num_gpus=parallel_config.world_size)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True,
                 runtime_env={"env_vars":{"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1"}})

    if parallel_config.placement_group:
        # Placement group is already set.
        return

    device_str = "NPU"
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 "
                    f"{device_str}.")
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group."
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}.")
    else:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # Log a warning message and delay resource allocation failure response.
        # Avoid immediate rejection to allow user-initiated placement group
        # created and wait cluster to be ready
        if parallel_config.world_size > num_devices_in_cluster:
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.", device_str,
                device_str)
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{
            device_str: 1.0
        } for _ in range(parallel_config.world_size)])

        # vLLM engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")
        # This way, at least bundle is required to be created in a current
        # node.
        placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, parallel_config, device_str)
    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group
