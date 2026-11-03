from core.tool_registry import ToolRegistry
from security.permissions import PermissionManager


def dummy_tool():
    return "success"


registry = ToolRegistry()

registry.register(
    name="echo",
    description="Test tool",
    function=dummy_tool,
    category="system"
)

permissions = PermissionManager()

tool = registry.get("echo")

assert permissions.check(tool) is True

permissions.block_tool("echo")

try:
    permissions.check(tool)
    raise AssertionError("Blocked tool was incorrectly allowed.")

except PermissionError:
    print("PASS: Blocked tool rejected.")

print("Permission Manager test completed.")