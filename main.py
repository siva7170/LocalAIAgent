from core.agent import Agent
from core.tool_registry import ToolRegistry
from security import workspace
from security.workspace import WorkspaceManager
from tools.filesystem.reader import FileReader, ReadFileArguments
from config import WORKSPACE_PATH

from tools.system.echo import (
    echo,
    EchoArguments
)

from tools.filesystem.directory import (
    DirectoryReader,
    ListDirectoryArguments
)
from tools.filesystem.scanner import (
    FileScanner,
    FindFilesArguments
)
from tools.filesystem.writer import (
    FileWriter,
    WriteFileArguments
)
from tools.filesystem.editor import (
    FileEditor,
    EditFileArguments
)
from tools.filesystem.appender import (
    FileAppender,
    AppendFileArguments
)
from tools.filesystem.deleter import (
    FileDeleter,
    DeleteFileArguments
)

def delete_test():
    print("[EXECUTION] Sensitive operation executed.")

    return {
        "status": "success"
    }


def create_registry():

    registry = ToolRegistry()

    registry.register(
        name="echo",
        description="Returns the supplied message. Use this to test tool execution.",
        function=echo,
        arguments_model=EchoArguments,
        category="system",
        requires_approval=False
    )

    workspace = WorkspaceManager(WORKSPACE_PATH)

    reader = FileReader(workspace)

    registry.register(
        name="read_file",
        description=(
            "Reads a text file inside the configured workspace. "
            "Use this to inspect source code, configuration files, "
            "logs and text documents."
        ),
        function=reader.read_file,
        arguments_model=ReadFileArguments,
        category="filesystem",
        requires_approval=False
    )

    directory_reader = DirectoryReader(workspace)

    registry.register(
        name="list_directory",
        description=(
            "Lists files and subdirectories inside the configured workspace. "
            "Use this to inspect directory contents."
        ),
        function=directory_reader.list_directory,
        arguments_model=ListDirectoryArguments,
        category="filesystem",
        requires_approval=False
    )

    file_scanner = FileScanner(workspace)

    registry.register(
        name="find_files",
        description=(
            "Searches for files inside the configured workspace "
            "using filename patterns or extensions. Supports "
            "recursive and non-recursive searches. Examples: "
            "*.txt, sample.txt, *.py"
        ),
        function=file_scanner.find_files,
        arguments_model=FindFilesArguments,
        category="filesystem",
        requires_approval=False
    )

    file_writer = FileWriter(workspace)

    registry.register(
        name="write_file",
        description=(
            "Creates a text file inside the configured workspace. "
            "Can overwrite an existing file only when overwrite=true. "
            "Use this to create source code, configuration files "
            "and text documents."
        ),
        function=file_writer.write_file,
        arguments_model=WriteFileArguments,
        category="filesystem",
        requires_approval=True
    )

    file_editor = FileEditor(workspace)

    registry.register(
        name="edit_file",
        description=(
            "Edits an existing text file inside the configured workspace "
            "by replacing exact text. Can replace the first occurrence "
            "or all occurrences. Requires user approval."
        ),
        function=file_editor.edit_file,
        arguments_model=EditFileArguments,
        category="filesystem",
        requires_approval=True
    )

    file_appender = FileAppender(workspace)

    registry.register(
        name="append_file",
        description=(
            "Appends text content to the end of an existing file "
            "inside the configured workspace. Does not overwrite "
            "existing content or create new files. Requires user approval."
        ),
        function=file_appender.append_file,
        arguments_model=AppendFileArguments,
        category="filesystem",
        requires_approval=True
    )

    file_deleter = FileDeleter(workspace)

    registry.register(
        name="delete_file",
        description=(
            "Deletes an existing file inside the configured workspace. "
            "Cannot delete directories or files outside the workspace. "
            "This is a destructive operation and requires user approval."
        ),
        function=file_deleter.delete_file,
        arguments_model=DeleteFileArguments,
        category="filesystem",
        requires_approval=True
    )

    registry.register(
        name="delete_test",
        description="Simulates a sensitive operation.",
        function=delete_test,
        category="system",
        requires_approval=True
    )

    return registry


def main():

    registry = create_registry()

    print("\nRegistered Tools:")
    for tool_name in registry.list_tools():
        print(f"- {tool_name}")

    agent = Agent(registry)

    print("\nLocal AI Agent Started")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        try:
            response = agent.run(question)
            print("\nAI:", response, "\n")

        except Exception as ex:
            print(f"\nAgent Error: {ex}\n")


if __name__ == "__main__":
    main()