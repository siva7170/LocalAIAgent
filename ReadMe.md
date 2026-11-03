powershell -ExecutionPolicy Bypass -NoExit -File .\.venv\Scripts\Activate.ps1

python -m tests.test_permissions

python -m tests.test_approval

python -m tests.test_workspace

python -m tests.test_audit