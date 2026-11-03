
import ollama
import json

from config import MODEL
from ai.prompts import RESPONDER_PROMPT


class Responder:

    def __init__(self):

        self.client = ollama.Client()

    def generate(self, question, history):

        response = self.client.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": RESPONDER_PROMPT
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "question": question,
                        "execution_results": history
                    })
                }
            ],
            options={"temperature": 0.3}
        )

        return response.message.content
