MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
MODEL_REVISION = "48d6d0fc4e02fb1269b36940650a1b7233035cbb"

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
#MODEL_REVISION = "7740ff69081bd553f4879f71eebcc2d6df2fbcb3"

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-FP8"
#MODEL_REVISION = "30cf496dde104e5309a8f7ed3ec33e7722b280f5"

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B"
#MODEL_REVISION = "156cc136479303ef79cb0aedf91e2f2e911d1c5c"

#MODEL_NAME = "mistralai/Mistral-7B-v0.3"
#MODEL_REVISION = "d8cadc02ac76bd617a919d50b092e59d2d110aff"

#MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
#MODEL_REVISION = "ffe1a706bacbd5abddc5ff99432ee38f7e0662fb"

#MODEL_NAME = "mistralai/Mixtral-8x22B-v0.1"
#MODEL_REVISION = "0cb34c4e8b821c03614475934f839447d46da342"

import modal

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(
		[
			"huggingface_hub",
			"hf-transfer",
		]
	)
	.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

@app.function(volumes={"/models": volume}, timeout=4 * 60 * 60)
def download_model(model_name, model_revision):
	import os

	from huggingface_hub import snapshot_download

	volume.reload()

	if not os.path.isdir(f"/models/{model_name}"):
		print(f"Starting download of {model_name}")

		os.makedirs(f"/models/{model_name}")

		snapshot_download(
			model_name,
			local_dir=f"/models/{model_name}",
			revision=model_revision,
			ignore_patterns=["*.pt", "*.bin", "*.pth", "original/*"],
		)

		print(f"Finished download of {model_name}")

		volume.commit()

		print("Saved data to disk")


@app.local_entrypoint()
def main():
	download_model.remote(MODEL_NAME, MODEL_REVISION)