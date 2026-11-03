from json import tool

from models.execution_result import ExecutionResult

from security.permissions import PermissionManager
from security.approval import ApprovalManager
from security.workspace import WorkspaceManager
from security.audit import SecurityAudit

from config import WORKSPACE_PATH


class Orchestrator:

    def __init__(self, registry):

        self.registry = registry

        self.permissions = PermissionManager()
        self.approvals = ApprovalManager()
        self.workspace = WorkspaceManager(WORKSPACE_PATH)
        self.audit = SecurityAudit()

    def execute_task(self, task):

        tool = self.registry.get(task.tool)

        # 1. Verify tool registration.
        if tool is None:

            self.audit.record(
                "TOOL_REJECTED",
                f"Unknown tool: {task.tool}"
            )

            return ExecutionResult(
                task_id=task.id,
                tool=task.tool,
                success=False,
                error=f"Tool not registered: {task.tool}"
            )

        try:

            self.audit.record(
                "TOOL_REQUESTED",
                f"Tool: {tool.name}"
            )

            # 2. Check permissions.
            self.permissions.check(tool)

            self.audit.record(
                "PERMISSION_GRANTED",
                f"Tool: {tool.name}"
            )

            # 3. Validate arguments.
            arguments = self.registry.validate_arguments(
                task.tool,
                task.arguments
            )

            # Validate filesystem paths before requesting approval.
            if tool.category == "filesystem" and "path" in arguments:
            
                self.workspace.validate_path(
                    arguments["path"]
                )

            # 4. Request approval when required.
            # Always trust registered tool metadata, not Qwen's response.
            if tool.requires_approval:

                approved = self.approvals.request_approval(
                    tool,
                    arguments
                )

                if not approved:

                    self.audit.record(
                        "APPROVAL_DENIED",
                        f"Tool: {tool.name}"
                    )
                
                    return ExecutionResult(
                        task_id=task.id,
                        tool=task.tool,
                        success=False,
                        error="User denied operation approval.",
                        approval_denied=True
                    )

                self.audit.record(
                    "APPROVAL_GRANTED",
                    f"Tool: {tool.name}"
                )

            # 5. Execute the registered function.
            output = tool.function(**arguments)

            self.audit.record(
                "EXECUTION_SUCCESS",
                f"Tool: {tool.name}"
            )

            return ExecutionResult(
                task_id=task.id,
                tool=task.tool,
                success=True,
                output=output
            )

        except Exception as ex:

            self.audit.record(
                "EXECUTION_FAILED",
                f"Tool: {tool.name} | Error type: {type(ex).__name__}"
            )

            return ExecutionResult(
                task_id=task.id,
                tool=task.tool,
                success=False,
                error=str(ex)
            )

    def execute_plan(self, plan):

        results = []

        for task in plan.tasks:

            result = self.execute_task(task)
            results.append(result)

        return results