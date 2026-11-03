from ai.planner import Planner
from ai.responder import Responder

from core.context import AgentContext
from core.orchestrator import Orchestrator

from config import MAX_EXECUTION_ROUNDS, MAX_TASKS_PER_PLAN
from models import task


class Agent:

    def __init__(self, registry):

        self.registry = registry
        self.planner = Planner()
        self.responder = Responder()
        self.orchestrator = Orchestrator(registry)

    def run(self, question):

        context = AgentContext(question)
        completed = False

        for round_number in range(MAX_EXECUTION_ROUNDS):

            print(f"\n[PLANNING ROUND {round_number + 1}]")

            plan = self.planner.create_plan(
                question=context.original_question,
                history=context.get_history(),
                available_tools=self.registry.get_available_tools()
            )

            if len(plan.tasks) > MAX_TASKS_PER_PLAN:
                raise ValueError("Plan contains too many tasks.")

            print("[UNDERSTANDING]", plan.understanding)
            print("[TASK COUNT]", len(plan.tasks))
            print("[PLAN DONE]", plan.done)

            if plan.done and not plan.tasks:
                completed = True
                break

            if not plan.tasks:
                print("[WARNING] No tasks returned.")
                break

            # Prevent repeating an already successful operation.
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
                
                    print(
                        f"[SKIPPED] Task {task.id}: "
                        "Identical operation already succeeded."
                    )

                    repeated_task = True
                    continue
                
                print(
                    f"[TASK {task.id}] "
                    f"{task.tool}: {task.purpose}"
                )

                result = self.orchestrator.execute_task(task)

                context.add_result(result, task)

                print(
                    f"[RESULT] Task {result.task_id} | "
                    f"Tool: {result.tool} | "
                    f"Success: {result.success}"
                )

                if result.error:
                    print("[ERROR]", result.error)

                # Stop planning when a tool execution fails.
                if not result.success:

                    print(
                        "[STATUS] Tool execution failed. "
                        "Stopping the current request."
                    )
                    tool_failed = True
                    break

                if result.output is not None:
                    print("[OUTPUT]", result.output)

                # Stop immediately if the user denies approval.
                if result.approval_denied:
                
                    approval_denied = True

                    print(
                        "[STATUS] User denied approval. "
                        "Stopping the current request."
                    )

                    break
                
            if tool_failed:
                break

            if approval_denied:
                break

            if repeated_task:
                completed = True
                print("[STATUS] Previously successful operation detected.")
                break

            # if repeated_task:
            #     print(
            #         "[STATUS] Duplicate task skipped. "
            #         "Continuing to the next planning round."
            #     )
            #     continue

            if plan.done:
                completed = all(
                    result["success"]
                    for result in context.get_history()
                )
                break

        print("\n[GENERATING RESPONSE]")

        if not completed:
            print("[STATUS] Request was not confirmed as completed.")

        return self.responder.generate(
            question=context.original_question,
            history=context.get_history()
        )