import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import ollama

from config import (
    LLAMA_CPP,
    LLAMA_CPP_MODEL_PATH,
    LLAMA_CPP_N_CTX,
    LLAMA_CPP_N_GPU_LAYERS,
    MODEL,
    OLLAMA,
    OLLAMA_HOST,
    USE,
)


@lru_cache(maxsize=1)
def _get_ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


@lru_cache(maxsize=1)
def _get_llama_cpp_model() -> Any:
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except ImportError as ex:
        raise RuntimeError(
            "The LLAMA_CPP backend requires llama-cpp-python. "
            "Install it with: pip install -r requirements-llama-cpp.txt"
        ) from ex

    model_path = Path(LLAMA_CPP_MODEL_PATH).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"llama.cpp GGUF model file not found: {model_path.resolve()}"
        )

    return llama_cpp.Llama(
        model_path=str(model_path),
        n_ctx=LLAMA_CPP_N_CTX,
        n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
        verbose=False,
    )


def chat(
    messages: list[dict[str, str]],
    temperature: float,
    json_mode: bool = False,
) -> str:
    return "".join(stream_chat(messages, temperature, json_mode))


def stream_chat(
    messages: list[dict[str, str]],
    temperature: float,
    json_mode: bool = False,
) -> Iterator[str]:
    if USE == OLLAMA:
        request_options: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True,
        }
        if json_mode:
            request_options["format"] = "json"

        for response in _get_ollama_client().chat(**request_options):
            if response.message.content:
                yield response.message.content
        return

    if USE == LLAMA_CPP:
        options: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if json_mode:
            options["response_format"] = {"type": "json_object"}

        for completion in _get_llama_cpp_model().create_chat_completion(**options):
            for choice in completion.get("choices", []):
                content = (choice.get("delta") or {}).get("content", "")
                if content:
                    yield content
        return

    raise ValueError(
        f"Unsupported model backend {USE!r}; "
        f"choose {OLLAMA!r} or {LLAMA_CPP!r}."
    )
