import time

from ai.chat_backend import stream_chat
from ai.planner import Planner
from ai.responder import Responder
from config import MAX_EXECUTION_ROUNDS, MAX_TASKS_PER_PLAN
from core.context import AgentContext
from core.orchestrator import Orchestrator


class Agent:

    def __init__(self, registry, approval_callback=None):
        self.registry = registry
        self.planner = Planner()
        self.responder = Responder()
        self.orchestrator = Orchestrator(
            registry,
            approval_callback=approval_callback,
        )

    def run(self, question):
        response = []

        for event in self.run_stream(question):
            event_type = event["type"]

            if event_type == "stage_start" and event["stage"] == "planning":
                print(f"\n[PLANNING ROUND {event['round']}]")
            elif event_type == "understanding":
                print("[UNDERSTANDING]", event["text"])
            elif event_type == "plan":
                print("[TASK COUNT]", len(event["tasks"]))
                print("[PLAN DONE]", event["done"])
            elif event_type == "task_start":
                task = event["task"]
                print(f"[TASK {task['id']}] {task['tool']}: {task['purpose']}")
            elif event_type == "task_skipped":
                print(f"[SKIPPED] Task {event['task_id']}: {event['reason']}")
            elif event_type == "task_complete":
                print(
                    f"[RESULT] Task {event['task_id']} | "
                    f"Tool: {event['tool']} | Success: {event['success']}"
                )
                if event["error"]:
                    print("[ERROR]", event["error"])
                if event["output"] is not None:
                    print("[OUTPUT]", event["output"])
            elif event_type == "status":
                print("[STATUS]", event["message"])
            elif event_type == "stage_start" and event["stage"] == "response":
                print("\n[GENERATING RESPONSE]")
            elif event_type == "response_token":
                response.append(event["token"])

        return "".join(response)

    def run_stream(self, question, conversation_history=None):
        context = AgentContext(question)
        completed = False

        for round_number in range(MAX_EXECUTION_ROUNDS):
            round_id = round_number + 1
            planning_stage_id = f"planning-{round_id}"
            planning_started = time.perf_counter()
            yield {
                "type": "stage_start",
                "stage": "planning",
                "stage_id": planning_stage_id,
                "round": round_id,
            }

            messages = self.planner.build_messages(
                question=context.original_question,
                history=context.get_history(),
                available_tools=self.registry.get_available_tools(),
                conversation_history=conversation_history,
            )
            plan_tokens = []
            for token in stream_chat(
                messages,
                temperature=0.1,
                json_mode=True,
            ):
                plan_tokens.append(token)
                yield {
                    "type": "planning_token",
                    "stage_id": planning_stage_id,
                    "token": token,
                }

            plan = self.planner.parse_plan("".join(plan_tokens))
            if len(plan.tasks) > MAX_TASKS_PER_PLAN:
                raise ValueError("Plan contains too many tasks.")

            yield {
                "type": "understanding",
                "round": round_id,
                "text": plan.understanding,
            }
            yield {
                "type": "plan",
                "round": round_id,
                "done": plan.done,
                "tasks": [task.model_dump() for task in plan.tasks],
            }
            yield {
                "type": "stage_complete",
                "stage": "planning",
                "stage_id": planning_stage_id,
                "elapsed_ms": round(
                    (time.perf_counter() - planning_started) * 1000
                ),
            }

            if plan.done and not plan.tasks:
                completed = True
                break

            if not plan.tasks:
                yield {
                    "type": "status",
                    "message": "No tasks returned.",
                }
                break

            previous_successes = {
                (
                    item["tool"],
                    str(sorted(item.get("arguments", {}).items()))
                )
                for item in context.get_history()
                if item["success"]
            }

            repeated_task = False
            approval_denied = False
            tool_failed = False

            for task in plan.tasks:
                task_signature = (
                    task.tool,
                    str(sorted(task.arguments.items()))
                )

                if task_signature in previous_successes:
                    repeated_task = True
                    yield {
                        "type": "task_skipped",
                        "round": round_id,
                        "task_id": task.id,
                        "reason": "Identical operation already succeeded.",
                    }
                    continue

                task_stage_id = f"task-{round_id}-{task.id}"
                task_started = time.perf_counter()
                yield {
                    "type": "task_start",
                    "stage_id": task_stage_id,
                    "round": round_id,
                    "task": task.model_dump(),
                }

                result = self.orchestrator.execute_task(task)
                context.add_result(result, task)

                yield {
                    "type": "task_complete",
                    "stage_id": task_stage_id,
                    "round": round_id,
                    "elapsed_ms": round(
                        (time.perf_counter() - task_started) * 1000
                    ),
                    **result.model_dump(),
                }

                if result.approval_denied:
                    approval_denied = True
                    yield {
                        "type": "status",
                        "message": "User denied approval. Stopping this request.",
                    }
                    break

                if not result.success:
                    tool_failed = True
                    yield {
                        "type": "status",
                        "message": "Tool execution failed. Stopping this request.",
                    }
                    break

            if tool_failed or approval_denied:
                break

            if repeated_task:
                completed = True
                yield {
                    "type": "status",
                    "message": "Previously successful operation detected.",
                }
                break

            if plan.done:
                completed = all(
                    result["success"]
                    for result in context.get_history()
                )
                break

        response_stage_id = "response"
        response_started = time.perf_counter()
        yield {
            "type": "stage_start",
            "stage": "response",
            "stage_id": response_stage_id,
        }

        for token in self.responder.generate_stream(
            question=context.original_question,
            history=context.get_history(),
            conversation_history=conversation_history,
        ):
            yield {
                "type": "response_token",
                "stage_id": response_stage_id,
                "token": token,
            }

        yield {
            "type": "stage_complete",
            "stage": "response",
            "stage_id": response_stage_id,
            "elapsed_ms": round(
                (time.perf_counter() - response_started) * 1000
            ),
        }
        yield {
            "type": "run_complete",
            "completed": completed,
        }
