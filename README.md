# Content

The main source containing:
  - `angular_steering.ipynb`: analysis and visualization of activations and
    directions, extraction of the steering directions, construction of the steering
    plane.
  - `generate_responsees.py`: generation of steered responses (with our vLLM fork) used for evaluation.
  - `evaluate_jailbreak.py`: evaluation of `substring_matching`, `LlamaGuard 3`,
    `HarmBench` and LLM-as-a-judge.
  - `eval_perplexity.py`: evaluating the perplexity of steered generation.
  - `eval.sh`: script to run the evaluation of tinyBenchmarks. (with `github.com/EleutherAI/lm-evaluation-harness`)
  - `generate_random_config.py`: generation of steering configs on random planes.
  - `visualization.ipynb`: visualization of benchmark results used in the paper.
  - `endpoint.py`: an OpenAI-compatible endpoint server to play with Angular Steering.
  - `steering_demo.py`: a gradio chat UI that uses the endpoint server.

## NEW: vLLM v1 Engine Support (feat/vllm_steering branch)

This branch includes native vLLM v1 support using standard vLLM (no custom fork required):
  - **`vllm_angular_steering.py`**: Complete module with steering logic and batch generation CLI

Usage as library:
```python
from vllm import LLM
from vllm_angular_steering import AngularSteering

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", enforce_eager=True)
steering = AngularSteering(llm)
steering.load_config_from_file("steering_config.npy")
steering.apply_steering(target_degree=180, adaptive_mode=1)
```

Usage for batch generation:
```bash
python vllm_angular_steering.py --model Qwen/Qwen2.5-7B-Instruct \
    --config-dir output/Qwen2.5-7B-Instruct --language en --angle-step 10
```

Compatible with vLLM v0.6+ (requires `enforce_eager=True`). See module docstring for details.

You will also need
- Our fork of vLLM with the necessary modifications to support Angular Steering: https://github.com/lone17/vllm/ (for original implementation)
- **OR** Standard vLLM v0.6+ for the new `vllm_angular_steering.py` implementation

# Installation

1. Create conda environment:

```bash
conda create -n angular_steering python=3.10
conda activate angular_steering
cd llm-activation-control/
pip install -r requirements.txt
conda deactivate
cd ..
```

2. Install vLLM from source (following the instructions in the vLLM repo https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html):

```bash
cd ../vllm/
conda activate angular_steering
VLLM_USE_PRECOMPILED=1 pip install --editable .
conda deactivate
cd ..
```

3. Create a separated conda environment for lm-evaluation-harness and install it
   following https://github.com/EleutherAI/lm-evaluation-harness:

```bash
cd ..
conda create -n lm_eval python=3.10
conda activate lm_eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
conda deactivate
cd ..
```

# Reproducing the results

## Using precomputed results

We provide the precomputed outputs and results in the `output/` and `benchmarks/`
folders. You can directly use the `visualization.ipynb` notebook to visualize the
results and run the `endpoint.py` and `steering_demo.py` scripts to play with Angular
Steering.

## Generating the results

1. Run the `angular_steering.ipynb` notebook to extract the steering directions and
   construct the steering plane. This will generate the `output/` folder with the
   extracted steering directions and the steering plane.
2. Run the `generate_responsees.py` script to generate the steered responses.
3. The `evaluate_jailbreak.py` script will evaluate the steered responses with
   `substring_matching`, `LlamaGuard 3`, `HarmBench` and LLM-as-a-judge. Some of the
   evaluation requires serving LLMs beforehand.
   - Serve `meta-llama/Llama-Guard-3-8B` using vLLM on port 8898 for `LlamaGuard 3` evaluation.
   - Serve `Qwen/QVQ-72B-Preview` using vLLM on port 8809 for LLM-as-a-judge evaluation.
   - For `HarmBench`, the script will serve `cais/HarmBench-Llama-2-13b-cls` natively so
     make sure you have enough GPU memory to run it.
   - For `substring_matching`, it does not require serving any LLMs.
   - You should edit the `methods` list in the main function of the script to
     include/exclude the methods you want to evaluate.
   - The output will be saved in the `output/` folder.
4. Run the `eval_perplexity.py` script to evaluate the perplexity of the steered
   generation. The result will be saved in the `output/` folder.
5. Run the `eval.sh` script in the `lm-eval` environment to evaluate the steered
   generation on tinyBenchmarks.
   - You must first serve run `endpoint.py` script to serve an OpenAI-compatible endpoint
     server with Angular Steering supported. Change the model name in the script to the
     one you want to evaluate.
   - Change the `MODELS` and `port` variables in the `eval.sh` script to the models you want to
     evaluate and the port of the endpoint server.
   - The output will be saved in the `benchmarks/` folder.
   - This might take a while.
6. Run the `visualization.ipynb` notebook to visualize the benchmark results.
7. Optionally, you can run the `endpoint.py` and `steering_demo.py` script to play with Angular Steering
   using a gradio chat UI.

---

Thank you for your interest in our work!
