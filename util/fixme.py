#MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
#MODEL_PARAMETERS = 8

import modal

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(
		[
			"vllm==0.5.3post1",
			"transformers",
			"accelerate",
			"tensorizer",
		]
	)
)

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(image=image, secrets=[modal.Secret.from_name("my-huggingface-secret")])

@app.function(volumes={"/models": volume}, memory=((MODEL_PARAMETERS * 1024) * 10) / 4, timeout=4 * 60 * 60)
def tensorize_model(model_name, model_revision):
	import os

	from vllm import LLM
	from vllm.engine.arg_utils import EngineArgs
	from vllm.model_executor.model_loader.tensorizer import (TensorizerArgs,
															TensorizerConfig,
															tensorize_vllm_model)

	volume.reload()

	if not os.path.isdir(f"/models/{model_name}-Tensorized"):
		print(f"Starting tensorization of {model_name}")

		tensorizer_config = TensorizerConfig(tensorizer_uri=f"/models/{model_name}")

# see https://docs.vllm.ai/en/stable/getting_started/examples/tensorize_vllm_model.html
# see https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer
# see https://github.com/coreweave/tensorizer


		print(f"Finished tensorization of {model_name}")

		volume.commit()

		print("Saved data to disk")

@app.local_entrypoint()
def main():
	tensorize_model.remote(MODEL_NAME, MODEL_REVISION)