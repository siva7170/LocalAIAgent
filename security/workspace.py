from pathlib import Path


class WorkspaceManager:

    def __init__(self, workspace_path):

        self.root = Path(workspace_path).resolve()

        self.root.mkdir(
            parents=True,
            exist_ok=True
        )

    def validate_path(self, requested_path):

        requested = Path(requested_path)

        if not requested.is_absolute():
            requested = self.root / requested

        resolved = requested.resolve()

        try:
            resolved.relative_to(self.root)

        except ValueError:
            raise PermissionError(
                f"Access outside workspace is not allowed: {resolved}"
            )

        return resolved

    def get_root(self):

        return self.root