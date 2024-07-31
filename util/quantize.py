#MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
#MODEL_PARAMETERS = 8

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
#MODEL_PARAMETERS = 70

#MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B"
#MODEL_PARAMETERS = 405

#MODEL_NAME = "mistralai/Mistral-7B-v0.3"
#MODEL_PARAMETERS = 7

#MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
#MODEL_PARAMETERS = 47

#MODEL_NAME = "mistralai/Mixtral-8x22B-v0.1"
#MODEL_PARAMETERS = 141

import modal

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(
		[
			"sentencepiece",
		]
	)
	.apt_install("git")
	.run_commands("git clone https://github.com/neuralmagic/AutoFP8.git && pip install -e AutoFP8")
)

volume = modal.Volume.from_name("models")

app = modal.App(image=image)

@app.function(volumes={"/models": volume}, memory=(MODEL_PARAMETERS * 1024) + 4096, timeout=4 * 60 * 60)
def quantize_model(model_name):
	import os

	from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

	volume.reload()

	if not os.path.isdir(f"/models/{model_name}-Dynamic"):
		print(f"Starting quantization of {model_name}")

		quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")

		model = AutoFP8ForCausalLM.from_pretrained(f"/models/{model_name}", quantize_config)
		model.quantize([])
		model.save_quantized(f"/models/{model_name}-Dynamic")

		print(f"Finished quantization of {model_name}")

		volume.commit()

		print("Saved data to disk")


@app.local_entrypoint()
def main():
	quantize_model.remote(MODEL_NAME)