import vllm_mindspore # Add this line on the top of script.

from vllm import LLM, SamplingParams
'''
from . import set_env
env_manager = set_env.EnvVarManager()
# def env
env_vars = {
    "ASCEND_CUSTOM_PATH": os.path.expandvars("$ASCEND_HOME_PATH/../"),
    "MS_ENABLE_LCCL": "off",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1",
    "MS_ALLOC_CONF": "enable_vmm:True",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0"
}
'''


# Sample prompts.
prompts = [
    "Hey, are you conscious? Can you talk to me?",
    "What will talk to the last human being?"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)

# Create an LLM.
# llm = LLM(model="/home/hyx/models/Qwen/Qwen2.5-0.5B-Instruct")
# llm = LLM(model="/home/hyx/models/Qwen/Qwen2.5_var-0.5B-Instruct")
llm = LLM(model="/home/hyx/models/Qwen/Qwen2.5-7B-Instruct")
# llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
