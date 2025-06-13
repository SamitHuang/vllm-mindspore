import vllm_mindspore # Add this line on the top of script.
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams


def main(args):
    model_path = args.model_path
    processor = AutoProcessor.from_pretrained(model_path)
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
    image = Image.open(args.image_path)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)

    # Create an LLM.
    llm = LLM(model=model_path,
            max_model_len=32768,
            max_num_seqs=8,
            tensor_parallel_size=args.tp_size,
            )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    query = {"prompt": text, "multi_modal_data": {"image": image}}
    inputs = [query for i in range(args.batch_size)] 
    outputs = llm.generate(inputs, sampling_params)

    # Print the outputs.
    print("========== output =========")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
    print("==========================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="/home/mikecheung/model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_path", type=str, default="demo.jpeg")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--tp_size", type=int, default=1)
    args, _ = parser.parse_known_args()

    main(args)
