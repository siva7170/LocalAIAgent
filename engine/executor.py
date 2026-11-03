
from tools.system.powershell import execute


def execute_plan(plan):

    results = []

    for task in plan["tasks"]:

        task_id = task.get("id")
        action = task.get("action")
        purpose = task.get("purpose", "")
        command = task.get("command", "")

        print(f"\n[AI TASK {task_id}] {purpose}")

        if action != "powershell":
            result = {
                "success": False,
                "error": f"Unsupported action: {action}"
            }

        else:
            try:
                result = execute(command)

            except Exception as ex:
                result = {
                    "success": False,
                    "error": str(ex)
                }

        print("\n[COMMAND RESULT]")
        print(result)

        results.append({
            "task_id": task_id,
            "action": action,
            "purpose": purpose,
            "command": command,
            "result": result
        })

    return results
