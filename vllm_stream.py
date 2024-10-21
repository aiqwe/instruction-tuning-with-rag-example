import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

# lora apdapter 다운로드
lora_path = snapshot_download(repo_id="aiqwe/gemma-2b-it-example-v1")

# AsyncLLMEngine 설정
engine_args = AsyncEngineArgs(model="google/gemma-2b-it", enforce_eager=True, enable_lora=True)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Streaming 함수 설정
async def generate_streaming(prompt):

    sampling_params = SamplingParams(
    temperature=0,
    max_tokens=512,
    stop=["[/assistant]"]
    )
    
    results_generator = engine.generate(prompt, sampling_params, lora_request=LoRARequest("my_model", 1, lora_path), request_id = "unique_0")
    
    async for request_output in results_generator:
        print(request_output.outputs[0].text)

prompts = "전세 계약에 대해 알려줘요."

asyncio.run(generate_streaming(prompts))