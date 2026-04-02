from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


def main():
    engine = LLMEngine(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1,
        max_model_len=2048,
    )

    prompt = "What is the Transformer?"
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    print("Prompt:", prompt)
    print("\nOutput:\n", end="")

    for event in engine.generate_stream([prompt], sampling_params):
        if event["type"] == "token":
            print(event["delta_text"], end="", flush=True)
        elif event["type"] == "finished":
            print("\n")

    engine.exit()


if __name__ == "__main__":
    main()
