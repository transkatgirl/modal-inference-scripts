MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Dynamic"
MODEL_LEN = 32000

import modal

MEMORY_GB = 24
GPU_COUNT = 1
GPU_TYPE = modal.gpu.A100(count=GPU_COUNT, size="40GB")
MAX_CONCURRENCY = 128

IDLE_TIMEOUT = 2

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

@app.function(
    image=image,
    cpu=GPU_COUNT/2.0,
    gpu=GPU_TYPE,
    memory=(MEMORY_GB * 1024, (MEMORY_GB * 1024) + (4096 * GPU_COUNT)),
    container_idle_timeout=IDLE_TIMEOUT * 60,
    allow_concurrent_inputs=MAX_CONCURRENCY,
    timeout=int(MODEL_LEN/4) + 8,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("api-token")]
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

    engine_args = AsyncEngineArgs(
        model=f"/models/{MODEL_NAME}",
        max_model_len=MODEL_LEN,
        #trust_remote_code=True,
        tensor_parallel_size=GPU_COUNT,
        tokenizer_pool_size=int(GPU_COUNT/2)+4,
        gpu_memory_utilization=0.98,
        enforce_eager=True, # Reduces cold-start time and memory usage at the cost of performance
        enable_prefix_caching=True,
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