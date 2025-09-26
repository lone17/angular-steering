import os

os.environ["VLLM_USE_V1"] = "0"
import sys
import time
from functools import cache
from pathlib import Path
from traceback import print_exc
from typing import List

import uvicorn
import vllm  # assuming the vllm fork with control vectors is installed
from common.const import EXTRACTION_POINT_ID
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams


# New request model for OpenAI chat completions
class CompletionRequest(BaseModel):
    class Config:
        extra = "allow"

    model: str
    # messages: list[dict]  # each dict should have "role" and "content"
    prompt: str | list[str]
    max_tokens: int = 4000
    logprobs: int = 0
    echo: bool = False
    temperature: float = 0.0


def get_model(model_id):
    return vllm.LLM(
        model=model_id,
        enable_control_vector=True,
        max_control_vectors=1,
        max_seq_len_to_capture=8192,
        gpu_memory_utilization=0.8,
        # max_model_len=4096,
        # quantization="fp8",
    )


@cache
def get_steering_config_path(model_id, extraction_strategy_id):
    model_family, model_name = model_id.split("/")
    output_path = Path("outputs") / "thinking-steering-configs" / model_name
    extraction_point_id = EXTRACTION_POINT_ID[model_id][extraction_strategy_id]

    steering_config_path = (
        output_path
        / f"steering_config-{extraction_strategy_id}-{extraction_point_id}.npy"
    )

    if Path(steering_config_path).exists():
        return steering_config_path

    return None


# cuda 0 uvicorn endpoint:app --host 0.0.0.0 --port 9900
app = FastAPI()

LANGUAGE = "en"

# Get model_id from environment variable or command line argument
if len(sys.argv) > 1:
    model_id = sys.argv[1]
else:
    raise ValueError("Model ID must be provided as a command line argument.")

MODEL_PORTS = {
    "Qwen/Qwen3-4B": 9905,
    # "Qwen/Qwen3-8B": 9907,
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 9903,
    # "deepseek-ai/DeepSeek-R1-Distill-LLama-8B": 9907
}

# Get the port for the current model
if model_id not in MODEL_PORTS:
    raise ValueError(f"Model ID {model_id} is not recognized.")
port = MODEL_PORTS.get(model_id)

llm = get_model(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# New endpoint for chat completions compatible with OpenAI API
@app.post("/thinking_steering/{extraction_strategy_id}/{rotation_degree}")
async def create_completion(
    extraction_strategy_id, rotation_degree, request: CompletionRequest
):
    global model_id, llm

    requested_model_id = request.model

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    steering_config_path = get_steering_config_path(
        requested_model_id, extraction_strategy_id
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    cv_request = None
    if rotation_degree != "none":
        control_vector_name = f"{model_id}/{steering_config_path}/{rotation_degree}"
        control_vector_id = abs(hash((control_vector_name, rotation_degree))) % 999999
        cv_request = ControlVectorRequest(
            control_vector_name=control_vector_name,
            control_vector_id=control_vector_id,
            control_vector_local_path=str(steering_config_path),
            scale=10.0,
            target_degree=int(rotation_degree),
            keep_norm=False,
            adaptive_mode=1,
        )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Helper function to process a single chat request
async def process_single_chat(request: CompletionRequest):
    global model_id, llm
    requested_model_id, steering_id = request.model.rsplit("/", 1)
    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    class Config:
        extra = "allow"

    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: int = 4000
    temperature: float = 1.0
    top_p: float = 1.0


@app.post(
    "/thinking_steering/{extraction_strategy_id}/{rotation_degree}/v1/chat/completions"
)
async def create_chat_completion_with_steering(
    extraction_strategy_id: str, rotation_degree: str, request: ChatCompletionRequest
):
    global model_id, llm

    requested_model_id = request.model

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    steering_config_path = get_steering_config_path(
        requested_model_id, extraction_strategy_id
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    cv_request = None
    if rotation_degree != "none":
        control_vector_name = f"{model_id}/{steering_config_path}/{rotation_degree}"
        control_vector_id = abs(hash((control_vector_name, rotation_degree))) % 999999
        cv_request = ControlVectorRequest(
            control_vector_name=control_vector_name,
            control_vector_id=control_vector_id,
            control_vector_local_path=str(steering_config_path),
            scale=10.0,
            target_degree=int(rotation_degree),
            keep_norm=False,
            adaptive_mode=1,
        )

    # Convert the ChatCompletionRequest to a format suitable for llm.chat
    messages = []
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Prepare sampling parameters
    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("messages", None)
    sampling_params = SamplingParams(**__params)

    try:
        # Use the chat function instead of generate
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        # apply chat template and call generate
        # prompts = tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, tokenize=False
        # )

        # import pdb

        # pdb.set_trace()

        # outputs = llm.generate(
        #     prompts=prompts,
        #     sampling_params=sampling_params,
        #     control_vector_request=cv_request,
        # )

        responses = []
        for item in outputs:
            # Extract the assistant's response from the chat output
            text_output = item.outputs[0].text

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text_output},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(item.prompt_token_ids),
                        "completion_tokens": len(item.outputs[0].token_ids),
                        "total_tokens": (
                            len(item.prompt_token_ids) + len(item.outputs[0].token_ids)
                        ),
                    },
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Add this at the bottom of the file
if __name__ == "__main__":
    print(f"Starting server for model: {model_id} on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
