import vllm_mindspore # Add this line on the top of script.
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams

import mindspore as ms
ms.set_context(pynative_synchronize=True)

MODEL_NAME = "/home/mikecheung/model/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("========== input =========")
print(text)
print("==========================")


# Load image using PIL.Image
image = Image.open("demo.jpeg").resize((256, 256)).convert("RGB")

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)

# Create an LLM.
llm = LLM(model=MODEL_NAME, max_model_len=32768)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate({"prompt": text, "multi_modal_data": {"image": image}}, sampling_params)

# Print the outputs.
print("========== output =========")
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
print("==========================")
