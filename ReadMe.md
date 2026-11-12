powershell -ExecutionPolicy Bypass -NoExit -File .\.venv\Scripts\Activate.ps1

python -m tests.test_permissions

python -m tests.test_approval

python -m tests.test_workspace

python -m tests.test_audit

## Model backend

Ollama remains the default. In `config.py`, set `USE = OLLAMA` to use the
configured Ollama model and host.

To load a GGUF model directly in this Python process, install the optional
binding:

```text
pip install -r requirements-llama-cpp.txt
```

Set `USE = LLAMA_CPP` and point `LLAMA_CPP_MODEL_PATH` in `config.py` to your
Q4_K_M `.gguf` file. `LLAMA_CPP_N_CTX` controls context size, and
`LLAMA_CPP_N_GPU_LAYERS` controls GPU offload (`0` uses CPU only; use `-1` to
offload all layers when the installed build supports your GPU).

The GGUF model is loaded once and reused for planning and responses. The
llama.cpp Python binding contains native code; on Windows, installing it may
require a C++ build toolchain and CMake if a compatible wheel is unavailable.

## Web chat

Install the web dependencies with `pip install -r requirements.txt`, then run
`python main.py` and open `http://127.0.0.1:8000`. The page streams planning
tokens, the agent's understanding, tool stages and the final response. Tool
operations that require approval wait for a decision in the page. Run
`python main.py --cli` to use the original terminal interface.