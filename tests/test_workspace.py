from pathlib import Path

from security.workspace import WorkspaceManager
from config import WORKSPACE_PATH


workspace = WorkspaceManager(WORKSPACE_PATH)

print("Workspace:", workspace.get_root())


# Test 1: Access inside workspace.
try:

    path = workspace.validate_path("test.txt")

    print("PASS: Workspace path accepted:", path)

except PermissionError as ex:

    print("FAIL:", ex)


# Test 2: Reject access outside workspace.
try:

    workspace.validate_path("../outside.txt")

    raise AssertionError(
        "Outside workspace path was incorrectly accepted."
    )

except PermissionError:

    print("PASS: Outside workspace path rejected.")


# Test 3: Reject absolute path outside workspace.
try:

    workspace.validate_path(
        str(Path("C:/Windows/win.ini"))
    )

    raise AssertionError(
        "External absolute path was incorrectly accepted."
    )

except PermissionError:

    print("PASS: External absolute path rejected.")