import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class WebApiTests(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(main.app)

    def test_home_serves_single_page_chat_ui(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Local AI Agent", response.text)
        self.assertIn("Planning · round", response.text)

    def test_chat_endpoint_streams_server_sent_events(self):
        def fake_run(run, _question, _conversation_history):
            run.events.put({
                "type": "understanding",
                "text": "The user wants a greeting.",
            })
            run.events.put(None)

        with patch("main._run_chat", side_effect=fake_run):
            response = self.client.post(
                "/api/chat",
                json={"question": "Say hello."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"type": "understanding"', response.text)

    def test_approval_endpoint_resolves_waiting_tool_approval(self):
        run = main.ChatRun()
        result = []
        tool = SimpleNamespace(
            name="write_file",
            category="filesystem",
            description="Writes a file.",
        )
        worker = threading.Thread(
            target=lambda: result.append(
                run.request_approval(tool, {"path": "test.txt"})
            ),
            daemon=True,
        )
        worker.start()
        approval_event = run.events.get(timeout=2)

        response = self.client.post(
            "/api/approval",
            json={
                "request_id": approval_event["request_id"],
                "approved": True,
            },
        )
        worker.join(timeout=2)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(approval_event["type"], "approval_required")
        self.assertEqual(result, [True])


if __name__ == "__main__":
    unittest.main()
