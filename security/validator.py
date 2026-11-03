
import re

ALLOWED_COMMANDS = {
    "Get-Location",
    "Get-ChildItem",
    "Get-Item",
    "Test-Path",
    "Resolve-Path"
}

BLOCKED_CHARACTERS = [
    ";", "|", "&", ">", "<",
    "`", "$", "(", ")", "\n", "\r"
]


def validate_command(command):

    if not isinstance(command, str) or not command.strip():
        raise ValueError("Empty command.")

    if len(command) > 1000:
        raise ValueError("Command is too long.")

    for char in BLOCKED_CHARACTERS:
        if char in command:
            raise ValueError(
                f"Blocked character detected: {char}"
            )

    command_name = command.strip().split()[0]

    if command_name not in ALLOWED_COMMANDS:
        raise PermissionError(
            f"Command is not allowed: {command_name}"
        )

    return True
