from pydantic import BaseModel
from typing import Any


class ExecutionResult(BaseModel):

    task_id: int
    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    approval_denied: bool = False