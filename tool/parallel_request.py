import base64
from openai import OpenAI
from openai import AsyncOpenAI
import asyncio
import time


async def async_query_openai(query, model_path):
    aclient = AsyncOpenAI(
        base_url="http://0.0.0.0:9529/v1",
        api_key="EMPTY"
    )
    completion = await aclient.chat.completions.create(
        model=model_path,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0.,
        top_p=0.1,
        max_tokens=512
    )
    return completion.choices[0].message.content

async def async_process_queries(queries, model_path):
    results = await asyncio.gather(*(async_query_openai(query, model_path) for query in queries))
    return results

async def main(args):
    # single query
    image_path = args.image_path
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    image_base64 = f"data:image;base64,{encoded_image_text}"
    query = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64 
                    },
                },
                {"type": "text", "text": "What is shown in the image.?"},
            ]
    # multiple
    bs = args.batch_size
    queries = [query for i in range(bs)]

    start_time = time.time()
    results = await async_process_queries(queries, args.model_path)
    end_time = time.time()
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="/home/mikecheung/model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_path", type=str, default="/home/hyx/vllm/vllm-mindspore/demo.jpeg")
    parser.add_argument("--batch_size", type=int, default=2)
    args, _ = parser.parse_known_args()

    asyncio.run(main(args))
