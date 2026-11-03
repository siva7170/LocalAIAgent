
from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class AppendFileArguments(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=100000)


class FileAppender:

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def append_file(self, path: str, content: str):

        file_path = self.workspace.validate_path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path.name}"
            )

        if not file_path.is_file():
            raise ValueError(
                "The requested path is not a file."
            )

        with file_path.open(
            "a",
            encoding="utf-8",
            newline=""
        ) as file:
            file.write(content)

        return {
            "status": "success",
            "file": file_path.name,
            "path": str(file_path),
            "characters_appended": len(content)
        }
