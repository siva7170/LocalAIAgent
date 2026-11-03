
import ollama
import json

from config import MODEL, MAX_TASKS_PER_ROUND


SYSTEM_PROMPT = """
You are the planning engine of a local Windows AI assistant.

Understand the user's request and identify the operations
required to fulfil it.

Available action:
- powershell

Allowed PowerShell commands:
- Get-Location
- Get-ChildItem
- Get-Item
- Test-Path
- Resolve-Path

Return valid JSON only, using this structure:

{
    "understanding": "What the user wants",
    "done": false,
    "tasks": [
        {
            "id": 1,
            "action": "powershell",
            "purpose": "Why this command is needed",
            "command": "Get-Location"
        }
    ]
}

Rules:
1. Break the user's request into logical tasks.
2. Tasks must be ordered correctly.
3. Use previous execution results when planning subsequent tasks.
4. Do not repeat completed tasks.
5. If additional operations are required, create the next tasks.
6. If the request is fulfilled, set done to true and return an empty tasks list.
7. Never invent command results.
8. Do not create destructive or modifying commands.
9. Generate only commands from the allowed list.
10. Maximum tasks per response: """ + str(MAX_TASKS_PER_ROUND)


def create_plan(question, history):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": json.dumps({
                "original_question": question,
                "previous_execution_results": history,
                "instruction": (
                    "Plan the next required operations. "
                    "Use the previous results where necessary."
                )
            })
        }
    ]

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        format="json",
        options={
            "temperature": 0.1
        }
    )

    plan = json.loads(response.message.content)

    if not isinstance(plan, dict):
        raise ValueError("Invalid plan returned by model.")

    if not isinstance(plan.get("tasks"), list):
        raise ValueError("Plan must contain a tasks list.")

    if len(plan["tasks"]) > MAX_TASKS_PER_ROUND:
        raise ValueError("Too many tasks in plan.")

    return plan
