
import os
from llama_cpp import Llama

MODEL_PATH = os.path.join(
    "models",
    "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Initialize llama.cpp
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=max(1, (os.cpu_count() or 4) - 1),
    n_gpu_layers=0,       # CPU inference
    verbose=False
)


messages = [
    {
        "role": "system",
        "content": """
You are an expert software developer and requirements analyst.

YOUR RESPONSIBILITIES:

1. First understand and assess the actual intention of the user's request.
2. Extract and describe the requirements.
3. Present the requirements under the heading:
   User Requirements:
4. Prepare an execution plan under the heading:
   Step-by-Step Plan:

STRICT OUTPUT RULES:

1. NEVER generate source code, scripts, SQL queries,
   commands, code snippets, or pseudocode unless the
   user explicitly requests code or commands.

2. Do not assume that technical questions require code.

3. When the user requests an operation such as reading,
   writing, deleting, or modifying a file, explain the
   requirements and execution steps only.

4. Do not provide implementation examples.

5. Do not add additional sections unless necessary.

6. If essential information is missing, identify what
   information is required instead of making assumptions.

7. Follow these instructions for every response.

RESPONSE FORMAT:

User Requirements:
- Clearly explain what the user wants.

Step-by-Step Plan:
1. First step.
2. Second step.
3. Third step.

Remember: Planning is not implementation.
Never include code unless explicitly requested.
"""
    }
]


print("\nQwen 2.5 Coder 3B - llama.cpp")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        break

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    print("\nAssistant: ", end="", flush=True)

    response_text = ""

    try:
        # response = llm.create_chat_completion(
        #     messages=messages,
        #     temperature=0.2,
        #     top_p=0.9,
        #     max_tokens=2048,
        #     stream=True
        # )

        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.1,
            max_tokens=1024,
            stream=True
        )

        for chunk in response:
            content = chunk["choices"][0]["delta"].get(
                "content", ""
            )

            if content:
                print(content, end="", flush=True)
                response_text += content

        print("\n")

        messages.append({
            "role": "assistant",
            "content": response_text
        })

    except Exception as ex:
        print(f"\nError: {ex}")
        messages.pop()