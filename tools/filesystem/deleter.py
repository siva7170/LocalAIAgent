
from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class DeleteFileArguments(BaseModel):
    path: str = Field(min_length=1, max_length=500)


class FileDeleter:

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def delete_file(self, path: str):

        file_path = self.workspace.validate_path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path.name}"
            )

        if not file_path.is_file():
            raise ValueError(
                "The requested path is not a file."
            )

        file_name = file_path.name

        file_path.unlink()

        return {
            "status": "success",
            "file": file_name,
            "path": str(file_path),
            "deleted": True
        }
