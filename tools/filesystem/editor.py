
from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class EditFileArguments(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    search_text: str = Field(min_length=1, max_length=100000)
    replacement_text: str = Field(max_length=100000)
    replace_all: bool = False


class FileEditor:

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def edit_file(
        self,
        path: str,
        search_text: str,
        replacement_text: str,
        replace_all: bool = False
    ):

        file_path = self.workspace.validate_path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path.name}"
            )

        if not file_path.is_file():
            raise ValueError(
                "The requested path is not a file."
            )

        content = file_path.read_text(encoding="utf-8")

        occurrences = content.count(search_text)

        if occurrences == 0:
            raise ValueError(
                "Search text was not found in the file."
            )

        if replace_all:
            updated_content = content.replace(
                search_text,
                replacement_text
            )
            replacements = occurrences
        else:
            updated_content = content.replace(
                search_text,
                replacement_text,
                1
            )
            replacements = 1

        if updated_content == content:
            raise ValueError(
                "Replacement would not change the file."
            )

        file_path.write_text(
            updated_content,
            encoding="utf-8",
            newline=""
        )

        return {
            "status": "success",
            "file": file_path.name,
            "path": str(file_path),
            "replacements": replacements,
            "replace_all": replace_all,
            "characters_written": len(updated_content)
        }
