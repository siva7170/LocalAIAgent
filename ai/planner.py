import json
import ollama

from config import MODEL, OLLAMA_HOST
from models.plan import ExecutionPlan
from ai.prompts import PLANNER_PROMPT


class Planner:

    def __init__(self):

        self.client = ollama.Client(host=OLLAMA_HOST)

    def create_plan(self, question, history, available_tools):

        prompt = PLANNER_PROMPT.format(
            tools=json.dumps(available_tools, indent=2),
            history=json.dumps(history, indent=2),
            question=question
        )

        response = self.client.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0.1
            }
        )

        content = response["message"]["content"]

        plan_data = json.loads(content)

        return ExecutionPlan.model_validate(plan_data)