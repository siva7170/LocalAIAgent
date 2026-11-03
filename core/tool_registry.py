from dataclasses import dataclass
from typing import Any, Callable, Type
from pydantic import BaseModel


class EmptyArguments(BaseModel):
    pass


@dataclass
class ToolDefinition:
    name: str
    description: str
    function: Callable
    arguments_model: Type[BaseModel]
    category: str
    requires_approval: bool


class ToolRegistry:

    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        arguments_model: Type[BaseModel] = EmptyArguments,
        category: str = "general",
        requires_approval: bool = False
    ):

        if name in self.tools:
            raise ValueError(f"Tool already registered: {name}")

        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty.")

        if not callable(function):
            raise TypeError("Tool function must be callable.")

        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            function=function,
            arguments_model=arguments_model,
            category=category,
            requires_approval=requires_approval
        )

    def unregister(self, name: str):

        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")

        del self.tools[name]

    def get(self, name: str) -> ToolDefinition | None:
        return self.tools.get(name)

    def validate_arguments(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:

        tool = self.get(name)

        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        validated = tool.arguments_model.model_validate(arguments)

        return validated.model_dump()

    def get_available_tools(self) -> dict:

        available = {}

        for name, tool in self.tools.items():

            available[name] = {
                "description": tool.description,
                "category": tool.category,
                "requires_approval": tool.requires_approval,
                "arguments": tool.arguments_model.model_json_schema()
            }

        return available

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self.tools