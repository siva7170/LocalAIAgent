
import subprocess

from config import COMMAND_TIMEOUT, MAX_OUTPUT_LENGTH
from security.validator import validate_command


def execute(command):

    # Validate before executing
    validate_command(command)

    print("\n[EXECUTOR] Running PowerShell command:")
    print(">", command)

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command
        ],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT
    )

    output = result.stdout.strip()
    error = result.stderr.strip()

    if result.returncode != 0:
        return {
            "success": False,
            "exit_code": result.returncode,
            "output": output[:MAX_OUTPUT_LENGTH],
            "error": error[:MAX_OUTPUT_LENGTH]
        }

    return {
        "success": True,
        "exit_code": 0,
        "output": output[:MAX_OUTPUT_LENGTH],
        "error": error[:MAX_OUTPUT_LENGTH]
    }
