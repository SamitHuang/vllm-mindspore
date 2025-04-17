# Try to connect MindONE to vllm-mindspore

Maybe an easy way to connect mindone to vllm-ms. 

Core idea is to re-use the Attention class in vllm-mindspore, which features the PagedAttention, FA, KV cache, and prefill-decode branch forard, etc.

## versions

Based on vllm-mindspore docker 0411, update
- vllm-mindspore, master branch, 42df17c36c1f5
    - add changes in this repo

- mindone: https://github.com/SamitHuang/mindone/tree/vllm_qwen

## running

run_vllm_mindone.sh

```shell
export ASCEND_RT_VISIBLE_DEVICES=3,4

# set path to your mindone and vllm-mindspore
export PYTHONPATH=/home/hyx/vllm/vllm-mindspore:$PYTHONPATH
export PYTHONPATH=/home/hyx/vllm/mindone:$PYTHONPATH

# use MindONE backend
export vLLM_MODEL_BACKEND=MindOne

python test_vllm.py

```

test_vllm.py

```python
import vllm_mindspore # Add this line on the top of script.
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hey, are you conscious? Can you talk to me?",
    "What will talk to the last human being?"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)

# Create an LLM.
llm = LLM(model="/home/hyx/models/Qwen/Qwen2.5-7B-Instruct")
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Result on qwen2.5 7b:

bs=2, max_tokens=256, time cost 9s, ~ 57tokens/s

## TODOs
- [] in mindone/transformers/models/qwen/modeling_qwen_vllm.py, re-use more layers from mindone's implmentation, except for Qwen2Attention
- [] Performance comparsion to MF backend. same infer_boost and jit_level

## Add a New Model (a proposal)

Pick one existing model that is the most similar the new model. Take `qwen2vl.py` as example.

- In vllm-mindspore/model_executors/models/mo_models
    ``` 
    cp qwen2.py qwewn2vl.py
    ```

    call the Qwen2VLModel network from mindone, and change the lm head in this model, etc.

- In mindone/transformers/models/qwen2vl
    
    ```
    cp ../qwen2/modeling_qwen2_vllm.py modeling_qwen2vl_vllm.py
    ```

    Implement Qwen2VLModel based on vllm-ms Attention (self-attention), re-use the code in mindone modeling_qwen2vl.py as much as possible.


# vllm-mindspore

## Overview

The `vllm-mindspore`is a integration for running vLLM on the MindSpore framework.

This  is the recommended solution for supporting the MindSpore  within the vLLM community. It provides deep integration with the MindSpore framework, offering efficient computation and optimization support for vLLM, enabling seamless operation on MindSpore.

By using the `vllm-mindspore`, popular open-source models, can run seamlessly for training and inference on the MindSpore framework.

---

## Prerequisites

- Hardware: Atlas A2/A3
- Software:
    - Python >= 3.9
    - CANN >= 8.0.0
    - MindSpore >=2.5.0

---

## Getting Started

### Installation

#### Installation from source code

Install from source code. [Wiki Installation.](https://gitee.com/mindspore/vllm-mindspore/wikis/Getting%20Started/Installation)

#### Set up using Docker

##### Pre-built images

```shell
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:v1.0
```

##### Build image from source

```shell
docker build --network=host .
```

### Inference and Serving

#### Offline Inference

You can run vllm_mindspore in your own code on a list of prompts.

```bash
export ASCEND_TOTAL_MEMORY_GB=64 # Based on the ascend device.
```

```python

import vllm_mindspore # Add this line on the top of script.

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "I am",
    "Today is",
    "What is"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=8)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```

#### Serving（OpenAI-Compatible）

You can start the server via the vllm_mindspore command:

`python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-32B-Instruct" --tensor_parallel_size=8`

To call the server, you can use `curl` or any other HTTP client.

```shell

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "prompt": "MindSpore is",
    "max_tokens": 120,
    "temperature": 0
  }'

```

## Contributing

We welcome and value any contributions and collaborations:

- Please feel free comments about your usage of vllm_mindspore.
- Please let us know if you encounter a bug by filing an issue.

## License

Apache License 2.0, as found in the [LICENSE](https://gitee.com/mindspore/vllm_mindspore/blob/master/LICENSE) file.
