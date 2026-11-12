import asyncio
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.agent import Agent
from core.tool_registry import ToolRegistry
from security.workspace import WorkspaceManager
from tools.filesystem.reader import FileReader, ReadFileArguments
from config import USE, WORKSPACE_PATH

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

APPROVAL_TIMEOUT_SECONDS = 300
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
_pending_approvals = {}
_pending_approvals_lock = threading.Lock()


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list
    )


class ApprovalDecision(BaseModel):
    request_id: str
    approved: bool


class PendingApproval:

    def __init__(self):
        self.event = threading.Event()
        self.approved = False


class ChatRun:

    def __init__(self):
        self.events = queue.Queue()

    def request_approval(self, tool, arguments):
        request_id = uuid4().hex
        pending = PendingApproval()

        with _pending_approvals_lock:
            _pending_approvals[request_id] = pending

        self.events.put({
            "type": "approval_required",
            "request_id": request_id,
            "tool": {
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
            },
            "arguments": arguments,
        })

        try:
            if not pending.event.wait(timeout=APPROVAL_TIMEOUT_SECONDS):
                raise TimeoutError("Tool approval timed out.")
            return pending.approved
        finally:
            with _pending_approvals_lock:
                _pending_approvals.pop(request_id, None)


def _run_chat(run, question, conversation_history):
    try:
        agent = Agent(
            create_registry(),
            approval_callback=run.request_approval,
        )
        for event in agent.run_stream(
            question,
            conversation_history=conversation_history,
        ):
            run.events.put(event)
    except Exception as ex:
        run.events.put({
            "type": "error",
            "message": str(ex),
        })
    finally:
        run.events.put(None)


app = FastAPI(title="Local AI Agent")


@app.get("/", response_class=HTMLResponse)
async def home():
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": USE}


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    run = ChatRun()
    conversation_history = [
        message.model_dump()
        for message in payload.conversation_history
    ]
    worker = threading.Thread(
        target=_run_chat,
        args=(run, question, conversation_history),
        daemon=True,
    )
    worker.start()

    async def stream_events():
        while True:
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.to_thread(
                    run.events.get,
                    True,
                    0.25,
                )
            except queue.Empty:
                continue

            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/approval")
async def submit_approval(decision: ApprovalDecision):
    with _pending_approvals_lock:
        pending = _pending_approvals.get(decision.request_id)
        if pending is None:
            raise HTTPException(
                status_code=404,
                detail="Approval request not found or expired.",
            )
        if pending.event.is_set():
            raise HTTPException(
                status_code=409,
                detail="Approval request was already answered.",
            )
        pending.approved = decision.approved
        pending.event.set()

    return {"accepted": True}


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
    if "--cli" in sys.argv[1:]:
        main()
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)