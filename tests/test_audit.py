from pathlib import Path

from security.audit import SecurityAudit


audit = SecurityAudit()

audit.record(
    "TOOL_REGISTERED",
    "echo"
)

audit.record(
    "PERMISSION_GRANTED",
    "echo"
)

audit.record(
    "EXECUTION_SUCCESS",
    "echo"
)

audit.record(
    "APPROVAL_DENIED",
    "delete_test"
)

log_file = Path("logs/security.log")

assert log_file.exists(), "Audit log file was not created."

content = log_file.read_text(encoding="utf-8")

assert "TOOL_REGISTERED" in content
assert "PERMISSION_GRANTED" in content
assert "EXECUTION_SUCCESS" in content
assert "APPROVAL_DENIED" in content

print("PASS: Security audit file created.")
print("PASS: Security events recorded.")
print("Log file:", log_file.resolve())