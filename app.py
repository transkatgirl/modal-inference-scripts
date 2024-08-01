# TODO: Implement tensorizer for non-quantized models

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
#MODEL_VRAM = 80
#MODEL_LEN = 128000

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
#MODEL_VRAM = 320
#MODEL_LEN = 128000

MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-FP8"
MODEL_VRAM = 640
MODEL_LEN = 44000 # lower than context len of 128000 to reduce vRAM usage

#MODEL_NAME = "mistralai/Mistral-7B-v0.3"
#MODEL_VRAM = 24
#MODEL_LEN = 32000

#MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
#MODEL_VRAM = 160
#MODEL_LEN = 32000

# FIXME: Seems to produce broken outputs
#MODEL_NAME = "mistralai/Mixtral-8x22B-v0.1"
#MODEL_VRAM = 320
#MODEL_LEN = 48000 # lower than context len of 64000 to reduce vRAM usage

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "vllm==0.5.3post1",
        ]
    )
    .env({"VLLM_NO_USAGE_STATS": "1"})
)

volume = modal.Volume.from_name("models")

MODEL_ID = MODEL_NAME.split("/")[-1]

app = modal.App(f"vLLM.{MODEL_ID}", image=image)

import math

GPU_MEMORY = MODEL_VRAM * 1024

if GPU_MEMORY > (80 * 1024):
    GPU_COUNT = math.ceil(GPU_MEMORY / (80 * 1024))
else:
    GPU_COUNT = 1

if GPU_MEMORY > (40 * 1024):
    GPU_TYPE = modal.gpu.A100(count=GPU_COUNT, size="80GB")
elif GPU_MEMORY > (24 * 1024):
    GPU_TYPE = modal.gpu.A100(count=GPU_COUNT, size="40GB")
else:
    GPU_TYPE = "A10G"

CPU_MEMORY = GPU_MEMORY + (4096 * GPU_COUNT)

@app.function(
    image=image,
    cpu=max(GPU_COUNT/2.0, GPU_COUNT-4.0), # Only reserve half of each CPU core (up to 4 fewer cores), as usage can temporarily spike above this. See: https://modal.com/docs/guide/resources#reserving-cpu-and-memory
    gpu=GPU_TYPE,
    memory=(min(CPU_MEMORY, 344064), min(CPU_MEMORY, 344064)), # Containers currently have a hard limit of 344064 MB of memory
    timeout=3 * 60 * 60,
    container_idle_timeout=15 * 60,
    allow_concurrent_inputs=128,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("vllm-authentication-token")]
)
@modal.asgi_app()
def serve():
    import os

    import asyncio
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.usage.usage_lib import UsageContext

    app = fastapi.FastAPI(
        title=f"{MODEL_NAME} server",
        description="vLLM OpenAI-compatible LLM server",
        version="0.0.1",
        docs_url="/docs",
    )

    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token", description="Token is required for authentication"
    )

    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != os.environ["API_TOKEN"]:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])
    router.include_router(api_server.router)

    app.include_router(router)

    # TODO: As demand increases, optimize for throughput
    # https://docs.vllm.ai/en/latest/models/spec_decode.html
    # https://docs.vllm.ai/en/latest/models/performance.html

    import random

    engine_args = AsyncEngineArgs(
        model=f"/models/{MODEL_NAME}",
        max_model_len=MODEL_LEN,
        trust_remote_code=True,
        tensor_parallel_size=GPU_COUNT,
        tokenizer_pool_size=max(GPU_COUNT, 2),
        gpu_memory_utilization=0.98,
        enforce_eager=True,
        enable_prefix_caching=True,
        seed=random.randint(0, (2**32)-1),
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config=asyncio.run(engine.get_model_config()),
        served_model_names=[MODEL_NAME],
        lora_modules=[],
        prompt_adapters=[],
        request_logger=None,
    )

    return app