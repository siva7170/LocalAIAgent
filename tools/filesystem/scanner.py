from pathlib import Path
from pydantic import BaseModel, Field

from security.workspace import WorkspaceManager


class FindFilesArguments(BaseModel):

    path: str = Field(
        default=".",
        min_length=1,
        max_length=500
    )

    pattern: str = Field(
        min_length=1,
        max_length=200
    )

    recursive: bool = True

    max_results: int = Field(
        default=100,
        ge=1,
        le=1000
    )


class FileScanner:

    def __init__(self, workspace: WorkspaceManager):

        self.workspace = workspace

    def find_files(
        self,
        pattern: str,
        path: str = ".",
        recursive: bool = True,
        max_results: int = 100
    ):

        directory = self.workspace.validate_path(path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {path}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Path is not a directory: {path}"
            )

        if recursive:
            candidates = directory.rglob(pattern)
        else:
            candidates = directory.glob(pattern)

        results = []

        for item in candidates:

            # Only return files.
            if not item.is_file():
                continue

            # Validate every discovered path.
            safe_path = self.workspace.validate_path(item)

            results.append({
                "name": safe_path.name,
                "path": str(safe_path),
                "extension": safe_path.suffix
            })

            if len(results) >= max_results:
                break

        return {
            "search_directory": str(directory),
            "pattern": pattern,
            "recursive": recursive,
            "count": len(results),
            "results": results
        }