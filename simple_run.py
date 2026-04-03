import os

from transformers import AutoTokenizer

from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


def get_tensor_parallel_size():
    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def main():
    path = os.path.expanduser("~/model/Qwen3-32B/")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    tensor_parallel_size = get_tensor_parallel_size()
    engine = LLMEngine(
        model=path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=2048,
    )

    prompt = "What is the Transformer?"
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
    )

    print("Prompt:", prompt)
    print("\nOutput:\n", end="")

    import time

    t0 = time.time()
    e = next(engine.generate_stream([prompt], sampling_params))
    print(f"First token: {e['delta_text']}, time taken: {time.time() - t0:.2f} seconds")

    for event in engine.generate_stream([prompt], sampling_params):
        if event["type"] == "token":
            print(event["delta_text"], end="", flush=True)
        elif event["type"] == "finished":
            print("\n")
    t2 = time.time()
    print(f"Time taken: {t2 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
