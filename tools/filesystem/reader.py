from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class ReadFileArguments(BaseModel):

    path: str = Field(
        min_length=1,
        max_length=500
    )

    max_chars: int = Field(
        default=10000,
        ge=1,
        le=20000
    )


class FileReader:

    def __init__(self, workspace: WorkspaceManager):

        self.workspace = workspace

    def read_file(self, path: str, max_chars: int = 10000):

        file_path = self.workspace.validate_path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not file_path.is_file():
            raise ValueError(
                "The requested path is not a file."
            )

        with file_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read(max_chars + 1)

        truncated = len(content) > max_chars

        if truncated:
            content = content[:max_chars]

        return {
            "file": file_path.name,
            "content": content,
            "truncated": truncated,
            "characters_returned": len(content)
        }