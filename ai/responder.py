
import json

from ai.chat_backend import stream_chat
from ai.prompts import RESPONDER_PROMPT


class Responder:

    def generate_stream(
        self,
        question,
        history,
        conversation_history=None,
    ):
        return stream_chat(
            messages=self.build_messages(
                question,
                history,
                conversation_history,
            ),
            temperature=0.3,
        )

    def generate(self, question, history, conversation_history=None):
        return "".join(
            self.generate_stream(question, history, conversation_history)
        )

    def build_messages(self, question, history, conversation_history=None):
        return [
            {
                "role": "system",
                "content": RESPONDER_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps({
                    "question": question,
                    "execution_results": history,
                    "conversation_history": conversation_history or [],
                })
            }
        ]
