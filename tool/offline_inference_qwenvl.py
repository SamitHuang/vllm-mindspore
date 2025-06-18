import vllm_mindspore  # Add this line on the top of script.
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams


def prepare_text(processor: AutoProcessor, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "demo.jpeg",
                },
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text


def main(args):
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(model_path)

    prompts = ["Describe this photo.", "Guess where and when this photo was taken."]

    texts = [prepare_text(processor, prompt) for prompt in prompts]

    # Load image using PIL.Image
    image = Image.open(args.image_path)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

    # Create an LLM.
    llm = LLM(
        model=model_path,
        max_model_len=32768,
        max_num_seqs=8,
        tensor_parallel_size=args.tp_size,
    )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    inputs = [
        {"prompt": texts[i], "multi_modal_data": {"image": image}}
        for i in range(len(texts))
    ]
    outputs = llm.generate(inputs, sampling_params)

    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/mikecheung/model/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--image_path", type=str, default="demo.jpeg")
    parser.add_argument("--tp_size", type=int, default=1)
    args, _ = parser.parse_known_args()

    main(args)
