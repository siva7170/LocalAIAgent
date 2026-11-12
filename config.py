
MODEL = "qwen2.5-coder:7b"

OLLAMA_HOST = "http://localhost:11434"

# Select the local model backend.
OLLAMA = "OLLAMA"
LLAMA_CPP = "LLAMA.CPP"
USE = LLAMA_CPP

# LLAMA_CPP_MODEL_PATH = r".\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf"
LLAMA_CPP_MODEL_PATH = r".\models\qwen2.5-coder-3b-instruct-q4_k_m.gguf"
LLAMA_CPP_N_CTX = 4096
LLAMA_CPP_N_GPU_LAYERS = 0

# Maximum number of planning/execution rounds
MAX_EXECUTION_ROUNDS = 10

# Maximum tasks allowed in a single plan
MAX_TASKS_PER_PLAN = 10

# Maximum time for one PowerShell command
COMMAND_TIMEOUT = 60

# Maximum command output returned to the model
MAX_OUTPUT_LENGTH = 20000

WORKSPACE_PATH = "./workspaces"