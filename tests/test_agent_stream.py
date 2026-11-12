import unittest
from unittest.mock import Mock, patch

from core.agent import Agent
from models.execution_result import ExecutionResult
from models.plan import ExecutionPlan
from models.task import Task


class AgentStreamTests(unittest.TestCase):

    def test_run_stream_emits_plan_task_timing_and_response_events(self):
        registry = Mock()
        registry.get_available_tools.return_value = {}
        task = Task(
            id=1,
            tool="echo",
            purpose="Echo a test message.",
            arguments={"message": "hello"},
        )
        plan = ExecutionPlan(
            understanding="The user wants a greeting.",
            done=True,
            tasks=[task],
        )

        with patch("core.agent.Orchestrator") as orchestrator_type:
            agent = Agent(registry)

        agent.planner = Mock()
        agent.responder = Mock()
        agent.planner.build_messages.return_value = []
        agent.planner.parse_plan.return_value = plan
        agent.responder.generate_stream.return_value = iter(["Hello", "!"])
        orchestrator_type.return_value.execute_task.return_value = ExecutionResult(
            task_id=1,
            tool="echo",
            success=True,
            output="hello",
        )

        with patch("core.agent.stream_chat", return_value=iter(["{}"])):
            events = list(agent.run_stream("Say hello"))

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "stage_start")
        self.assertIn("planning_token", event_types)
        self.assertIn("understanding", event_types)
        self.assertIn("plan", event_types)
        self.assertIn("task_start", event_types)
        self.assertIn("task_complete", event_types)
        self.assertEqual(event_types[-1], "run_complete")

        completed_stages = [
            event for event in events
            if event["type"] in ("stage_complete", "task_complete")
        ]
        self.assertTrue(all(event["elapsed_ms"] >= 0 for event in completed_stages))


if __name__ == "__main__":
    unittest.main()
