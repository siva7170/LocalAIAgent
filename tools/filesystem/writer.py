from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class WriteFileArguments(BaseModel):

    path: str = Field(
        min_length=1,
        max_length=500
    )

    content: str = Field(
        max_length=100000
    )

    overwrite: bool = False


class FileWriter:

    def __init__(self, workspace: WorkspaceManager):

        self.workspace = workspace

    def write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False
    ):

        file_path = self.workspace.validate_path(path)

        if file_path.exists():

            if not file_path.is_file():
                raise ValueError(
                    "The requested path is not a file."
                )

            if not overwrite:
                raise FileExistsError(
                    f"File already exists: {file_path.name}"
                )

        # Do not automatically create missing parent directories.
        if not file_path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory not found: {file_path.parent}"
            )

        if not file_path.parent.is_dir():
            raise ValueError(
                "Parent path is not a directory."
            )

        # Exclusive creation prevents accidental overwriting.
        # New files must always use exclusive creation.
        # Existing files can only be overwritten when explicitly allowed.
        if file_path.exists():
        
            if not overwrite:
                raise FileExistsError(
                    f"File already exists: {file_path.name}"
                )

            mode = "w"

        else:
            mode = "x"

        with file_path.open(
            mode,
            encoding="utf-8",
            newline=""
        ) as file:

            file.write(content)

        return {
            "status": "success",
            "file": file_path.name,
            "path": str(file_path),
            "characters_written": len(content),
            "overwritten": mode == "w"
        }