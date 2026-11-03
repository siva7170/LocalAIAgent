from core.tool_registry import ToolRegistry
from security.approval import ApprovalManager


def dummy_tool():
    return "Operation executed."


registry = ToolRegistry()

registry.register(
    name="delete_test",
    description="Simulates a sensitive operation.",
    function=dummy_tool,
    category="system",
    requires_approval=True
)

approval_manager = ApprovalManager()

tool = registry.get("delete_test")

approved = approval_manager.request_approval(
    tool,
    {}
)

if approved:
    print("PASS: Approval granted.")
else:
    print("PASS: Approval denied.")