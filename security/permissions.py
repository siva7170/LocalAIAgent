class PermissionManager:

    def __init__(self):

        self.allowed_categories = {
            "general",
            "system",
            "filesystem",
            "development",
            "testing",
            "documents",
            "database"
        }

        self.blocked_tools = set()

    def check(self, tool):

        if tool.name in self.blocked_tools:
            raise PermissionError(
                f"Tool is blocked: {tool.name}"
            )

        if tool.category not in self.allowed_categories:
            raise PermissionError(
                f"Tool category is not allowed: {tool.category}"
            )

        return True

    def block_tool(self, name):
        self.blocked_tools.add(name)

    def unblock_tool(self, name):
        self.blocked_tools.discard(name)