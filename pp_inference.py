# /workspace/split_qwen_cpp/pp_inference.py
from tensorrt_llm import LLM, SamplingParams

def main():
    llm = LLM(
        model="/workspace/models/Qwen2.5-3B-Instruct",
        pipeline_parallel_size=2,
        dtype="float16",
    )

    prompts = ["讲一下 transformer"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

    for output in llm.generate(prompts, sampling_params):
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

if __name__ == '__main__':
    main()