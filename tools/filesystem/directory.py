from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class ListDirectoryArguments(BaseModel):

    path: str = Field(
        default=".",
        min_length=1,
        max_length=500
    )


class DirectoryReader:

    def __init__(self, workspace: WorkspaceManager):

        self.workspace = workspace

    def list_directory(self, path: str = "."):

        directory = self.workspace.validate_path(path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {path}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Path is not a directory: {path}"
            )

        items = []

        for item in sorted(
            directory.iterdir(),
            key=lambda x: (not x.is_dir(), x.name.lower())
        ):

            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file"
            })

        return {
            "directory": str(directory),
            "count": len(items),
            "items": items
        }