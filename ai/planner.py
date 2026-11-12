import json

from ai.chat_backend import chat
from models.plan import ExecutionPlan
from ai.prompts import PLANNER_PROMPT


class Planner:

    def build_messages(
        self,
        question,
        history,
        available_tools,
        conversation_history=None,
    ):
        prompt = PLANNER_PROMPT.format(
            tools=json.dumps(available_tools, indent=2),
            history=json.dumps(history, indent=2),
            conversation=json.dumps(conversation_history or [], indent=2),
            question=question
        )

        return [{"role": "user", "content": prompt}]

    def parse_plan(self, content):
        plan_data = json.loads(content)
        return ExecutionPlan.model_validate(plan_data)

    def create_plan(
        self,
        question,
        history,
        available_tools,
        conversation_history=None,
    ):
        content = chat(
            messages=self.build_messages(
                question,
                history,
                available_tools,
                conversation_history,
            ),
            temperature=0.1,
            json_mode=True,
        )
        return self.parse_plan(content)