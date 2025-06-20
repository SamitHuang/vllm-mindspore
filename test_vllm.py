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
# llm = LLM(model="/home/hyx/models/Qwen/Qwen2.5-0.5B-Instruct")
llm = LLM(model="/home/hyx/models/Qwen/Qwen2.5-7B-Instruct")
# llm = LLM(model="/home/mikecheung/model/opt-125m")
# llm = LLM(model="/home/mikecheung/model/opt-350m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
