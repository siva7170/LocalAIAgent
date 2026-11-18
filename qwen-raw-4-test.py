import os
import json
import ast
import hashlib
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, deque
from llama_cpp import Llama
from pathlib import Path


# ============================================================
# 1. MODEL INITIALIZATION
#    Keep the existing Qwen 2.5 Coder 7B configuration.
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=max(1, (os.cpu_count() or 4) - 1),
    n_gpu_layers=0,
    verbose=False
)


# ============================================================
# 2. COMMON LLM FUNCTION
# ============================================================

def ask_llm(system_prompt, user_prompt, max_tokens=3072):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        top_p=0.8,
        repeat_penalty=1.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    content = response["choices"][0]["message"]["content"]

    try:
        result = json.loads(content)
    except json.JSONDecodeError as ex:
        raise ValueError(
            f"Invalid JSON returned by model: {ex}\n{content}"
        )

    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")

    return result


# ============================================================
# 3. GENERAL VALIDATION HELPERS
# ============================================================

def require_list(data, key, stage_name):
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"{stage_name}: '{key}' must be a JSON array."
        )
    return value


def validate_sequential_ids(items, id_field, label):
    ids = [item.get(id_field) for item in items]

    if any(not isinstance(value, int) for value in ids):
        raise ValueError(f"{label}: all {id_field} values must be integers.")

    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate {id_field} values found.")

    expected = list(range(1, len(items) + 1))
    if sorted(ids) != expected:
        raise ValueError(
            f"{label}: {id_field} values must be sequential from 1. "
            f"Expected {expected}, received {ids}."
        )


def validate_operation_dependencies(operations):
    operation_ids = {item["operation_id"] for item in operations}

    for operation in operations:
        dependencies = operation.get("depends_on", [])

        if not isinstance(dependencies, list):
            raise ValueError(
                f"Operation {operation['operation_id']}: "
                "'depends_on' must be an array."
            )

        if operation["operation_id"] in dependencies:
            raise ValueError(
                f"Operation {operation['operation_id']} cannot depend on itself."
            )

        unknown = set(dependencies) - operation_ids
        if unknown:
            raise ValueError(
                f"Operation {operation['operation_id']} has unknown "
                f"dependencies: {sorted(unknown)}"
            )

    # Detect circular operation dependencies.
    graph = {
        item["operation_id"]: item.get("depends_on", [])
        for item in operations
    }
    visiting = set()
    visited = set()

    def visit(node):
        if node in visiting:
            raise ValueError("Circular operation dependency detected.")
        if node in visited:
            return

        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for operation_id in graph:
        visit(operation_id)


# ============================================================
# 4. STEP 1 - REQUIREMENT ANALYZER AND OPERATION PLANNER
# ============================================================

def analyze_requirements(user_input):
    system_prompt = """
You are an expert requirements analyst, software architect,
and task planner for an intelligent AI execution agent.

YOUR RESPONSIBILITY:
Understand the user's actual intention and convert it into
a complete, structured set of requirements and logical operations.

REASONING RULES:

1. Understand the desired end result, not merely the verbs in the request.
2. Identify explicit requirements and necessary implicit requirements.
3. Preserve exact filenames, paths, values, filters, table names,
   content, and constraints supplied by the user.
4. Identify prerequisites and dependencies between operations.
5. Identify expected outputs that later operations may need.
6. Identify risks, assumptions, and actions requiring special care.
7. Do not invent details that the user did not provide.
8. Ask no questions in this stage. Record essential unknowns as assumptions
   or unresolved_inputs.
9. Do not generate Python, SQL, shell commands, or executable code.
10. Do not execute anything.
11. Return valid JSON only.

OPERATION GRANULARITY:

- Describe logical operations, not one operation for every word or verb.
- If a request is naturally one action, keep it as one operation.
  Example: "Create file X with content Y" can be one logical operation
  named "create_file_with_content".
- Do not prematurely group unrelated operations. Grouping is Step 2.
- Do not split a single atomic action into artificial create/write steps
  unless the distinction is needed for dependency, validation, or recovery.
- Include verification as an operation when it materially helps prove
  that the user's requirement was achieved.
- Use the minimum complete set of operations that satisfies the request.

OPERATION FIELDS:

- operation_id: sequential integer starting at 1
- requirement_ids: requirement IDs addressed by this operation
- description: clear action description
- action: concise machine-readable action name
- target: target resource, or null if not applicable
- inputs: object containing required input values
- expected_output: actual output or state expected from this operation
- depends_on: array of operation IDs
- risk_level: low, medium, or high
- reversible: boolean

REQUIRED JSON STRUCTURE:

{
  "user_intent": "What the user wants to achieve",
  "user_requirements": [
    {
      "requirement_id": 1,
      "description": "Requirement",
      "expected_result": "Observable expected result",
      "priority": "high"
    }
  ],
  "operations": [
    {
      "operation_id": 1,
      "requirement_ids": [1],
      "description": "Logical operation",
      "action": "action_name",
      "target": "target or null",
      "inputs": {},
      "expected_output": "Expected output",
      "depends_on": [],
      "risk_level": "low",
      "reversible": true
    }
  ],
  "constraints": [],
  "assumptions": [],
  "unresolved_inputs": [],
  "overall_expected_result": "How completion will be verified"
}

Return only the JSON object. Do not wrap it in Markdown.
"""

    result = ask_llm(
        system_prompt,
        user_input,
        max_tokens=3072
    )

    requirements = require_list(
        result, "user_requirements", "Requirement Analyzer"
    )
    operations = require_list(
        result, "operations", "Requirement Analyzer"
    )

    validate_sequential_ids(
        requirements, "requirement_id", "Requirements"
    )
    validate_sequential_ids(
        operations, "operation_id", "Operations"
    )
    validate_operation_dependencies(operations)

    requirement_ids = {item["requirement_id"] for item in requirements}

    for operation in operations:
        linked_ids = operation.get("requirement_ids", [])
        unknown = set(linked_ids) - requirement_ids
        if unknown:
            raise ValueError(
                f"Operation {operation['operation_id']} references "
                f"unknown requirements: {sorted(unknown)}"
            )

    return result


# ============================================================
# 5. STEP 2 - PLAN ASSESSOR, CLASSIFIER AND OPTIMIZER
# ============================================================

def optimize_execution_plan(user_input, analysis):
    system_prompt = """
You are an expert AI workflow architect and execution-plan optimizer.

You receive:
- The original user request.
- User requirements.
- Atomic logical operations and their dependencies.

Your job is to assess all operations, explore possible execution
strategies, group operations intelligently, and select the best plan.

THIS IS A REASONING AND OPTIMIZATION STAGE.
Do not generate source code or execute operations.

A. OPERATION CLASSIFICATION

For every operation, select the most appropriate category:

- IO: file and directory operations
- DB: database-only operations
- IO_DB: file and database operations that form one logical workflow
- NETWORK: HTTP, API, network, or remote-service operations
- ANALYSIS: calculations, comparisons, interpretation, or analysis
- OTHER: operations that do not fit the above

Classification must reflect the actual work, not just keywords.

B. GROUPING RULES

1. Group operations that can efficiently and safely execute in one program.
2. Do not create a separate group merely because action names differ.
3. Related file operations should normally be grouped.
4. Related database operations may be grouped when transaction boundaries,
   dependencies, and recovery behavior make that appropriate.
5. Combine IO and DB operations as IO_DB when they form one logical workflow.
6. Do not combine unrelated operations just because they share a category.
7. Keep operations separate when they require different execution conditions,
   independent failure handling, different permissions, or meaningful
   checkpoints.
8. Respect all operation dependencies.
9. Operations in the same group may depend on earlier operations in that group.
10. A group must not depend on itself.
11. Avoid unnecessary groups and unnecessary execution boundaries.
12. Preserve outputs needed by later groups.

C. STRATEGY COMPARISON

Consider multiple strategies where useful, such as:
- Minimal groups / fewer execution boundaries
- Dependency-oriented groups
- Risk-isolated groups

Do not create artificial alternatives when only one sensible strategy exists.

Evaluate candidate strategies against:
- Requirement coverage and correctness
- Dependency correctness
- Execution efficiency
- Reliability and failure isolation
- Recoverability and checkpoint quality
- Avoidance of duplicate or repeated work

Select the strategy with the best overall balance.
Correctness and requirement coverage take priority over speed.

D. RECOVERY AWARENESS

For each group, explain:
- What output/state it should produce.
- What later groups may need from it.
- What should be checked before retrying if it fails.
- Whether the operation is safe to retry or may have side effects.

Do not assume that a failed operation made no changes.

E. IMPORTANT

- Every operation must appear in exactly one selected execution group.
- Preserve operation IDs from the input.
- Do not silently omit requirements or operations.
- Do not invent user-provided values.
- Do not generate Python, SQL, or shell code.
- Return valid JSON only.

REQUIRED JSON STRUCTURE:

{
  "operation_assessments": [
    {
      "operation_id": 1,
      "operation_type": "IO",
      "reason": "Why this classification is appropriate",
      "can_combine_with": [],
      "dependency_notes": "Dependency assessment",
      "side_effects": [],
      "retry_safety": "safe | inspect_before_retry | unsafe_without_recovery"
    }
  ],
  "candidate_strategies": [
    {
      "strategy_id": "S1",
      "strategy_name": "Strategy name",
      "groups": [
        {
          "group_id": 1,
          "group_type": "IO",
          "operation_ids": [1],
          "depends_on": [],
          "rationale": "Why these operations belong together",
          "expected_result": "Observable group result",
          "outputs_to_preserve": [],
          "failure_checks": [],
          "retry_approach": "How to approach recovery"
        }
      ],
      "evaluation": {
        "correctness": 10,
        "efficiency": 10,
        "reliability": 10,
        "recoverability": 10,
        "overall_score": 10,
        "trade_offs": []
      }
    }
  ],
  "selected_strategy_id": "S1",
  "selection_reason": "Why this strategy is best",
  "optimization_summary": "Summary of grouping and optimization decisions",
  "execution_groups": [
    {
      "group_id": 1,
      "group_type": "IO",
      "operation_ids": [1],
      "depends_on": [],
      "rationale": "Why these operations belong together",
      "expected_result": "Observable group result",
      "outputs_to_preserve": [],
      "failure_checks": [],
      "retry_approach": "How to approach recovery"
    }
  ]
}

The execution_groups array must be an exact copy of the selected strategy's
groups, with the same operation IDs and dependencies.

Scores must be integers from 1 to 10. Use a reasoned score, not arbitrary
perfect scores. Return only JSON without Markdown.
"""

    user_prompt = f"""
ORIGINAL USER REQUEST:
{user_input}

REQUIREMENTS AND OPERATIONS:
{json.dumps(analysis, indent=2, ensure_ascii=False)}

Assess every operation, compare sensible execution strategies,
and return the best optimized execution plan.
"""

    result = ask_llm(
        system_prompt,
        user_prompt,
        max_tokens=4096
    )

    assessments = require_list(
        result, "operation_assessments", "Plan Optimizer"
    )
    strategies = require_list(
        result, "candidate_strategies", "Plan Optimizer"
    )
    groups = require_list(
        result, "execution_groups", "Plan Optimizer"
    )

    original_operation_ids = {
        item["operation_id"] for item in analysis["operations"]
    }

    assessed_ids = [item.get("operation_id") for item in assessments]
    if set(assessed_ids) != original_operation_ids:
        raise ValueError(
            "Plan Optimizer must assess every operation exactly once."
        )
    if len(assessed_ids) != len(set(assessed_ids)):
        raise ValueError("Plan Optimizer returned duplicate operation assessments.")

    if not strategies:
        raise ValueError("Plan Optimizer returned no candidate strategies.")

    selected_id = result.get("selected_strategy_id")
    selected = next(
        (item for item in strategies if item.get("strategy_id") == selected_id),
        None
    )
    if selected is None:
        raise ValueError(
            "selected_strategy_id does not match a candidate strategy."
        )

    selected_groups = selected.get("groups")
    if not isinstance(selected_groups, list):
        raise ValueError("Selected strategy must contain a groups array.")

    # The final execution_groups must match the selected candidate.
    if groups != selected_groups:
        raise ValueError(
            "execution_groups must exactly match the selected strategy groups."
        )

    validate_grouping(groups, analysis["operations"])

    return result


def validate_grouping(groups, operations):
    operation_ids = {
        operation["operation_id"] for operation in operations
    }

    group_ids = [group.get("group_id") for group in groups]
    if any(not isinstance(group_id, int) for group_id in group_ids):
        raise ValueError("Every group_id must be an integer.")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Duplicate group_id values found.")

    grouped_operation_ids = []
    operation_to_group = {}

    for group in groups:
        group_operation_ids = group.get("operation_ids")

        if not isinstance(group_operation_ids, list) or not group_operation_ids:
            raise ValueError(
                f"Group {group.get('group_id')} must contain operation_ids."
            )

        for operation_id in group_operation_ids:
            if operation_id not in operation_ids:
                raise ValueError(
                    f"Group {group.get('group_id')} references unknown "
                    f"operation {operation_id}."
                )
            grouped_operation_ids.append(operation_id)
            operation_to_group[operation_id] = group["group_id"]

        if not isinstance(group.get("depends_on", []), list):
            raise ValueError(
                f"Group {group['group_id']}: depends_on must be an array."
            )

    if len(grouped_operation_ids) != len(set(grouped_operation_ids)):
        raise ValueError("An operation appears in more than one execution group.")

    if set(grouped_operation_ids) != operation_ids:
        missing = operation_ids - set(grouped_operation_ids)
        raise ValueError(
            f"Execution groups do not cover all operations. Missing: {sorted(missing)}"
        )

    group_id_set = set(group_ids)
    group_dependencies = {
        group["group_id"]: set(group.get("depends_on", []))
        for group in groups
    }

    for group_id, dependencies in group_dependencies.items():
        if group_id in dependencies:
            raise ValueError(f"Group {group_id} cannot depend on itself.")

        unknown = dependencies - group_id_set
        if unknown:
            raise ValueError(
                f"Group {group_id} has unknown dependencies: {sorted(unknown)}"
            )

    # Every cross-group operation dependency must be represented
    # by a dependency between the corresponding execution groups.
    for operation in operations:
        current_group = operation_to_group[operation["operation_id"]]

        for dependency_id in operation.get("depends_on", []):
            dependency_group = operation_to_group[dependency_id]

            if dependency_group != current_group:
                if dependency_group not in group_dependencies[current_group]:
                    raise ValueError(
                        f"Group {current_group} must depend on group "
                        f"{dependency_group}, because operation "
                        f"{operation['operation_id']} depends on operation "
                        f"{dependency_id}."
                    )

    # Detect circular group dependencies.
    visiting = set()
    visited = set()

    def visit(group_id):
        if group_id in visiting:
            raise ValueError("Circular execution-group dependency detected.")
        if group_id in visited:
            return

        visiting.add(group_id)
        for dependency in group_dependencies[group_id]:
            visit(dependency)
        visiting.remove(group_id)
        visited.add(group_id)

    for group_id in group_dependencies:
        visit(group_id)




# ============================================================
# 6. STEP 3 - GROUP CODE GENERATOR, VALIDATOR AND EXECUTOR
# ============================================================

GENERATED_DIR = Path("generated_scripts")
WORKSPACE_DIR = Path("workspaces")
RESULT_FILE = Path("agent_result.json")
GROUP_TIMEOUT_SECONDS = 120


def topological_group_order(groups):
    """Return groups in dependency order; reject invalid dependency graphs."""
    by_id = {group["group_id"]: group for group in groups}
    indegree = {group_id: 0 for group_id in by_id}
    dependents = defaultdict(list)

    for group_id, group in by_id.items():
        for dependency in group.get("depends_on", []):
            if dependency not in by_id:
                raise ValueError(
                    f"Group {group_id} depends on unknown group {dependency}."
                )
            indegree[group_id] += 1
            dependents[dependency].append(group_id)

    ready = deque(group_id for group_id in by_id if indegree[group_id] == 0)
    ordered = []
    while ready:
        group_id = ready.popleft()
        ordered.append(by_id[group_id])
        for child in dependents[group_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(ordered) != len(groups):
        raise ValueError("Circular execution-group dependency detected.")
    return ordered


def group_has_file_output(group, analysis):
    """Return True when this group includes a file-creation/write operation."""
    operations = {
        item["operation_id"]: item for item in analysis.get("operations", [])
    }
    for operation_id in group.get("operation_ids", []):
        operation = operations.get(operation_id, {})
        action = str(operation.get("action", "")).lower()
        inputs = operation.get("inputs", {})
        if "file" in action or "filename" in inputs or "file_path" in inputs:
            return True
    return False


def select_failure_injection(groups, analysis):
    """Offer a one-run, controlled failure test. Normal mode remains the default."""
    print("\n[CONTROLLED FAILURE-INJECTION TEST]")
    print("  A - Missing JSON import (NameError after the file is created)")
    print("  B - Failure after the file is created but before group output is saved")
    print("  C - Fail one dependency group; skip its dependents; continue independent groups")
    print("  N - Normal execution (no injected failure)")
    choice = input("Select test mode [A/B/C/N]: ").strip().upper() or "N"

    if choice == "N":
        return None

    if choice in {"A", "B"}:
        candidates = [g for g in groups if group_has_file_output(g, analysis)]
        if not candidates:
            print("[TEST MODE UNAVAILABLE] This plan has no identifiable file-output group.")
            return None
        target = candidates[0]
        print(f"[TEST MODE] Scenario {choice} will target group {target['group_id']}.")
        return {"mode": choice, "target_group_id": target["group_id"]}

    if choice == "C":
        by_id = {g["group_id"]: g for g in groups}
        dependents = defaultdict(list)
        parents = defaultdict(list)
        for group in groups:
            for dependency in group.get("depends_on", []):
                dependents[dependency].append(group["group_id"])
                parents[group["group_id"]].append(dependency)

        def closure(start, graph):
            seen = set()
            stack = list(graph.get(start, []))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(graph.get(current, []))
            return seen

        for target_id in by_id:
            downstream = closure(target_id, dependents)
            upstream = closure(target_id, parents)
            independent = [
                gid for gid in by_id
                if gid != target_id and gid not in downstream and gid not in upstream
            ]
            if downstream and independent:
                print(
                    f"[TEST MODE] Scenario C will fail group {target_id}; "
                    f"dependent groups {sorted(downstream)} should be skipped; "
                    f"independent groups {sorted(independent)} should continue."
                )
                return {
                    "mode": "C",
                    "target_group_id": target_id,
                    "expected_dependent_group_ids": sorted(downstream),
                    "expected_independent_group_ids": sorted(independent)
                }

        print(
            "[TEST MODE UNAVAILABLE] Scenario C needs an execution plan with "
            "at least one dependency chain and one independent group. "
            "Use a request that produces at least three separate groups, "
            "including one dependent operation and one unrelated operation."
        )
        return None

    print("[TEST MODE] Unknown selection. Continuing in normal mode.")
    return None


def inject_controlled_failure(code, mode):
    """Apply a deterministic test fault to generated source for the first attempt only."""
    import re

    if mode == "C":
        return (
            "# CONTROLLED TEST FAILURE: dependency-group failure\n"
            "raise RuntimeError('Injected dependency-group failure for Scenario C')\n\n"
            + code
        )

    if mode == "A":
        # Remove JSON imports from both the executor preamble and model output.
        code = re.sub(r"(?m)^\s*import\s+json(?:\s+as\s+\w+)?\s*(?:#.*)?$\n?", "", code)
        code = re.sub(r"(?m)^\s*from\s+json\s+import\s+.*$\n?", "", code)
        marker = "# CONTROLLED TEST FAILURE: missing json import\n"
        probe = "json.dumps({'injected_test': True})  # expected NameError\n"
        return insert_after_primary_file_write(code, marker + probe)

    if mode == "B":
        marker = "# CONTROLLED TEST FAILURE: after file creation, before group output\n"
        fault = "raise RuntimeError('Injected failure after primary file creation')\n"
        return insert_after_primary_file_write(code, marker + fault)

    return code


def insert_after_primary_file_write(code, injected_lines):
    """Insert a fault after the first top-level statement that writes file_path."""
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)

    def writes_primary_file(node):
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            function = child.func
            if isinstance(function, ast.Name) and function.id == "open":
                if child.args and isinstance(child.args[0], ast.Name) and child.args[0].id == "file_path":
                    return True
            if isinstance(function, ast.Attribute) and function.attr in {"write_text", "write_bytes"}:
                receiver = function.value
                if isinstance(receiver, ast.Name) and receiver.id == "file_path":
                    return True
    for statement in tree.body:
        if writes_primary_file(statement):
            insert_at = statement.end_lineno
            lines.insert(insert_at, injected_lines)
            return "".join(lines)

    raise ValueError(
        "Cannot inject this scenario safely: generated program has no recognizable "
        "primary file_path write operation."
    )


def generate_group_program(user_request, analysis, optimization, group,
                           previous_results, recovery_context=None):
    """Ask Qwen 7B for one complete Python program for this execution group."""
    operations_by_id = {
        item["operation_id"]: item for item in analysis["operations"]
    }
    group_operations = [
        operations_by_id[operation_id]
        for operation_id in group["operation_ids"]
    ]

    system_prompt = """
You are the Code Generator in a local AI execution engine.
Generate ONE complete, runnable Python program for ONE execution group.
Return a JSON object only, with exactly these fields:
{
  "program_name": "short descriptive name",
  "python_code": "complete Python source code",
  "implementation_notes": ["brief notes"],
  "expected_observations": ["what successful execution should print"]
}

Rules:
- Implement every operation in the supplied group, in dependency order.
- Do not implement operations belonging to other groups.
- Use only Python standard-library modules unless the user explicitly
  requested an installed package. Do not install packages.
- The executor injects WORKSPACE_DIR as a pathlib.Path and imports json,
  sys, and Path. Do not redefine WORKSPACE_DIR. Use it for task files.
  Create it if needed. Use paths built from WORKSPACE_DIR.
- Print clear progress and final results to stdout. Print errors to stderr
  and exit non-zero when an operation cannot be completed.
- Preserve useful outputs in a JSON file named group_output.json inside the
  supplied workspace. Include output values needed by dependent groups.
- Read prior group outputs only from the supplied previous_group_results.
  Do not fabricate missing values or claim an operation succeeded unless
  the code verifies it.
- Do not use subprocess, shell commands, eval, exec, dynamic imports,
  network access, credentials, or destructive cleanup.
- Do not delete or overwrite unrelated user files. If a target already
  exists, follow the user's stated overwrite intent; otherwise fail safely.
- For database work, use only connection details explicitly supplied by the
  user. Never invent credentials, hostnames, schemas, or database paths.
  If essential connection details are absent, report that clearly and exit
  non-zero rather than guessing.
- Avoid hardcoded sample data unless the user supplied it.
- If recovery_context is provided, treat it as the actual failure evidence.
  Diagnose and repair the existing attempt; do not repeat side effects that
  have already succeeded. Before creating a file/table/record, inspect whether
  the intended result already exists and matches the requested state. If it
  already matches, preserve it and perform only the remaining work.
- Do not blindly repeat a non-idempotent operation. Prefer a minimal repair.
- Return syntactically valid Python source as a JSON string.
- For Windows paths, use pathlib.Path with path components, or use forward
  slashes in string literals. Never write ordinary string literals containing
  backslash sequences such as a backslash followed by W; use raw strings only when
  appropriate. Avoid invalid-escape SyntaxWarnings.
"""

    context = {
        "original_user_request": user_request,
        "user_intent": analysis.get("user_intent"),
        "constraints": analysis.get("constraints", []),
        "assumptions": analysis.get("assumptions", []),
        "unresolved_inputs": analysis.get("unresolved_inputs", []),
        "execution_group": group,
        "operations_for_this_group": group_operations,
        "previous_group_results": previous_results,
        "recovery_context": recovery_context or {},
        "workspace_directory": str(WORKSPACE_DIR.resolve()),
        "group_output_contract": {
            "file": "group_output.json",
            "purpose": "Persist verified outputs required by later groups"
        }
    }

    response = ask_llm(
        system_prompt,
        json.dumps(context, indent=2, ensure_ascii=False),
        max_tokens=4096
    )

    code = response.get("python_code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Code Generator returned an empty python_code field.")

    return response


def validate_generated_program(python_code, script_path):
    """Static checks only; this is not a security sandbox."""
    if not python_code or not python_code.strip():
        raise ValueError("Generated program is empty.")

    try:
        tree = ast.parse(python_code, filename=str(script_path))
    except SyntaxError as ex:
        raise ValueError(f"Python syntax validation failed: {ex}") from ex

    blocked_imports = {
        "subprocess", "socket", "ctypes", "multiprocessing",
        "pty", "telnetlib", "ftplib", "paramiko", "requests"
    }
    blocked_calls = {"eval", "exec", "compile", "__import__", "breakpoint"}
    # Do not block generic methods such as list.remove().
    # Block only known destructive/system APIs by their receiver/module.
    blocked_os_methods = {"system", "popen", "remove", "rmdir"}
    blocked_path_methods = {"unlink", "rmdir"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked_imports:
                    raise ValueError(f"Blocked import detected: {root}")

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in blocked_imports:
                raise ValueError(f"Blocked import detected: {root}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
                raise ValueError(f"Blocked function call detected: {node.func.id}()")
            if isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                method = node.func.attr

                if (
                    isinstance(receiver, ast.Name)
                    and receiver.id == "os"
                    and method in blocked_os_methods
                ):
                    raise ValueError(
                        f"Blocked OS operation detected: os.{method}()"
                    )

                if (
                    isinstance(receiver, ast.Name)
                    and receiver.id == "shutil"
                    and method in {"rmtree", "move"}
                ):
                    raise ValueError(
                        f"Blocked filesystem operation detected: shutil.{method}()"
                    )

                if (
                    method in blocked_path_methods
                    and isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Name)
                    and receiver.func.id == "Path"
                ):
                    # Direct pathlib deletion calls are rejected in Step 3
                    # until recovery-aware deletion is implemented.
                    raise ValueError(
                        f"Blocked pathlib operation detected: Path(...).{method}()"
                    )

    return True


def save_agent_result(result):
    RESULT_FILE.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )


def execute_group_attempt(user_request, analysis, optimization, group,
                          previous_results, execution_state,
                          recovery_context=None, attempt_number=1,
                          failure_injection=None):
    group_id = group["group_id"]
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[GROUP {group_id}] Generating one complete Python program...")
    generated = generate_group_program(
        user_request, analysis, optimization, group, previous_results,
        recovery_context=recovery_context
    )

    script_path = GENERATED_DIR / f"group_{group_id:03d}_attempt_{attempt_number:02d}.py"
    code = generated["python_code"].strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    # The executor, not the LLM, owns the workspace path and essential
    # runtime imports. This prevents malformed Windows path literals and
    # missing json/sys imports in generated programs.
    import re
    code = re.sub(
        r"(?m)^\s*WORKSPACE_DIR\s*=.*(?:\r?\n|$)",
        "",
        code
    )
    runtime_preamble = (
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"WORKSPACE_DIR = Path({WORKSPACE_DIR.resolve().as_posix()!r})\n"
        "\n"
    )
    code = runtime_preamble + code.lstrip()

    if (
        failure_injection
        and failure_injection.get("target_group_id") == group_id
        and attempt_number == 1
    ):
        mode = failure_injection.get("mode")
        print(f"[TEST INJECTION] Applying controlled Scenario {mode} to group {group_id}.")
        code = inject_controlled_failure(code, mode)

    validate_generated_program(code, script_path)
    script_path.write_text(code + "\n", encoding="utf-8")

    print(f"[CODE GENERATED] {script_path}")
    print("[IMPLEMENTATION NOTES]")
    for note in generated.get("implementation_notes", []):
        print(f"  - {note}")
    print("\n" + "-" * 72)
    print(code)
    print("-" * 72)

    approval = input(
        f"Approve execution of group {group_id}? Type 'yes' to execute: "
    ).strip().lower()
    if approval != "yes":
        return {
            "group_id": group_id,
            "status": "approval_denied",
            "script_path": str(script_path),
            "stdout": "",
            "stderr": "Execution was not approved by the user.",
            "returncode": None,
            "implementation_notes": generated.get("implementation_notes", [])
        }

    print(f"\n[EXECUTING GROUP {group_id}]")
    try:
        process = subprocess.run(
            [sys.executable, "-u", str(script_path.resolve())],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GROUP_TIMEOUT_SECONDS
        )
        status = "succeeded" if process.returncode == 0 else "failed"
        outcome = {
            "group_id": group_id,
            "status": status,
            "script_path": str(script_path),
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.returncode,
            "implementation_notes": generated.get("implementation_notes", [])
        }
        if process.stdout:
            print("[STDOUT]\n" + process.stdout)
        if process.stderr:
            print("[STDERR]\n" + process.stderr)
        return outcome

    except subprocess.TimeoutExpired as ex:
        return {
            "group_id": group_id,
            "status": "timeout",
            "script_path": str(script_path),
            "stdout": "",
            "stderr": str(ex),
            "returncode": None,
            "implementation_notes": generated.get("implementation_notes", [])
        }
    except Exception as ex:
        return {
            "group_id": group_id,
            "status": "failed",
            "script_path": str(script_path),
            "stdout": "",
            "stderr": f"{type(ex).__name__}: {ex}",
            "returncode": None,
            "implementation_notes": generated.get("implementation_notes", [])
        }


def snapshot_workspace():
    """Capture actual workspace file state for partial-side-effect diagnosis."""
    snapshot = {}
    if not WORKSPACE_DIR.exists():
        return snapshot

    for path in WORKSPACE_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshot[str(path.relative_to(WORKSPACE_DIR))] = {
                "size": path.stat().st_size,
                "sha256": digest
            }
        except OSError as ex:
            snapshot[str(path.relative_to(WORKSPACE_DIR))] = {
                "inspection_error": f"{type(ex).__name__}: {ex}"
            }
    return snapshot


def compare_workspace_snapshots(before, after):
    changes = []
    all_paths = sorted(set(before) | set(after))
    for relative_path in all_paths:
        old = before.get(relative_path)
        new = after.get(relative_path)
        if old != new:
            if old is None:
                change_type = "created"
            elif new is None:
                change_type = "deleted"
            else:
                change_type = "modified"
            changes.append({
                "path": relative_path,
                "change_type": change_type,
                "before": old,
                "after": new
            })
    return changes


def diagnose_execution_failure(user_request, analysis, optimization, group,
                               attempt_result, workspace_changes,
                               recovery_history):
    """Ask Qwen to diagnose evidence and select a conservative recovery action."""
    system_prompt = """
You are the Result Analyzer and Recovery Planner in a local AI execution engine.
Analyze a failed execution using only the supplied evidence.
Do not claim that an operation succeeded unless evidence proves it.

Choose exactly one recovery_action:
- repair: generate a corrected complete program for this same group.
  Use this when code, imports, paths, output persistence, or another local
  implementation defect caused the failure. Preserve successful side effects.
- retry: repeat the same group only when the evidence shows the attempt made
  no side effects and repeating it is safe. Set retry_safe=true only then.
- stop: recovery is unsafe, impossible, requires missing user input, or the
  failure is not safely recoverable.

Never choose retry merely because an operation is marked reversible.
Consider stdout, stderr, exit code, changed workspace files, operation risk,
previous attempts, and the original user requirement. A created/modified file
is a real side effect even if the process later exits with an error.
If a requested output already exists and its contents are verified, tell the
Code Generator to preserve it and complete only the missing work.
Do not invent file contents, database state, or success evidence.

Return JSON only with this exact structure:
{
  "recovery_action": "repair|retry|stop",
  "diagnosis": "specific root cause based on evidence",
  "evidence": ["observed facts"],
  "partial_success": ["verified side effects, if any"],
  "remaining_work": ["work still needed"],
  "retry_safe": false,
  "repair_instructions": ["specific instructions for corrected program"],
  "user_input_required": [],
  "confidence": "low|medium|high"
}
"""
    payload = {
        "original_user_request": user_request,
        "requirements": analysis.get("user_requirements", []),
        "operations": [
            op for op in analysis.get("operations", [])
            if op.get("operation_id") in group.get("operation_ids", [])
        ],
        "execution_group": group,
        "failed_attempt": attempt_result,
        "workspace_changes_observed_by_python": workspace_changes,
        "previous_recovery_attempts": recovery_history,
    }
    result = ask_llm(
        system_prompt,
        json.dumps(payload, indent=2, ensure_ascii=False),
        max_tokens=2048
    )
    action = result.get("recovery_action")
    if action not in {"repair", "retry", "stop"}:
        raise ValueError(f"Recovery Planner returned invalid action: {action!r}")
    if not isinstance(result.get("retry_safe"), bool):
        raise ValueError("Recovery Planner must return boolean retry_safe.")
    for key in ("evidence", "partial_success", "remaining_work",
                "repair_instructions", "user_input_required"):
        if not isinstance(result.get(key), list):
            raise ValueError(f"Recovery Planner field '{key}' must be an array.")
    return result


def execute_group_program(user_request, analysis, optimization, group,
                          previous_results, execution_state,
                          failure_injection=None):
    """Execute, diagnose and recover a group with a bounded retry budget."""
    max_attempts = 3
    recovery_history = []
    recovery_context = None

    for attempt_number in range(1, max_attempts + 1):
        before = snapshot_workspace()
        outcome = execute_group_attempt(
            user_request, analysis, optimization, group, previous_results,
            execution_state, recovery_context=recovery_context,
            attempt_number=attempt_number,
            failure_injection=failure_injection
        )
        after = snapshot_workspace()
        workspace_changes = compare_workspace_snapshots(before, after)
        outcome["attempt_number"] = attempt_number
        outcome["workspace_changes"] = workspace_changes

        if outcome.get("status") == "succeeded":
            outcome["recovery_history"] = recovery_history
            return outcome

        if (
            failure_injection
            and failure_injection.get("mode") == "C"
            and failure_injection.get("target_group_id") == group["group_id"]
            and attempt_number == 1
        ):
            outcome["injected_failure"] = True
            outcome["recovery_history"] = recovery_history
            outcome["recovery_stopped"] = True
            outcome["recovery_stop_reason"] = (
                "Scenario C deliberately leaves this group failed to test "
                "dependency skipping and independent-group continuation."
            )
            print("[TEST MODE] Keeping this injected group failure so dependency handling can be verified.")
            return outcome

        if attempt_number >= max_attempts:
            outcome["recovery_history"] = recovery_history
            outcome["recovery_exhausted"] = True
            return outcome

        print(f"\n[STEP 4] Diagnosing failure in group {group['group_id']}...")
        try:
            diagnosis = diagnose_execution_failure(
                user_request, analysis, optimization, group, outcome,
                workspace_changes, recovery_history
            )
        except Exception as ex:
            outcome["recovery_diagnosis_error"] = f"{type(ex).__name__}: {ex}"
            outcome["recovery_history"] = recovery_history
            return outcome

        print("[FAILURE DIAGNOSIS]")
        print(json.dumps(diagnosis, indent=4, ensure_ascii=False))
        recovery_history.append({
            "attempt_number": attempt_number,
            "diagnosis": diagnosis,
            "workspace_changes": workspace_changes
        })

        action = diagnosis["recovery_action"]
        if action == "stop":
            outcome["recovery_history"] = recovery_history
            outcome["recovery_stopped"] = True
            return outcome

        if action == "retry":
            # Python, not the LLM, enforces this gate. Any observed workspace
            # mutation means an unmodified retry is not allowed.
            if workspace_changes or not diagnosis["retry_safe"]:
                print("[RECOVERY BLOCKED] Retry is unsafe because side effects "
                      "were observed or retry_safe was not confirmed.")
                outcome["recovery_history"] = recovery_history
                outcome["recovery_stopped"] = True
                outcome["recovery_block_reason"] = (
                    "Unmodified retry requires no workspace changes and "
                    "retry_safe=true."
                )
                return outcome
            recovery_context = {
                "action": "retry",
                "diagnosis": diagnosis,
                "instruction": "Retry only because Python observed no workspace changes."
            }
            print("[RECOVERY] LLM selected a safe retry; no workspace changes were detected.")
            continue

        recovery_context = {
            "action": "repair",
            "diagnosis": diagnosis,
            "workspace_changes": workspace_changes,
            "instruction": (
                "Repair the failed program. Preserve all successful side effects. "
                "Do not repeat already-completed operations."
            )
        }
        print("[RECOVERY] Generating a repaired program. User approval is required again.")

    return {
        "group_id": group["group_id"],
        "status": "failed",
        "stderr": "Recovery loop ended unexpectedly.",
        "recovery_history": recovery_history
    }


def execute_optimized_plan(user_request, analysis, optimization, final_output,
                           failure_injection=None):
    """Execute groups in dependency order; isolate failures and continue independent work."""
    groups = topological_group_order(optimization["execution_groups"])
    results_by_group = {}
    operation_status = {}

    final_output["execution_status"] = "in_progress"
    final_output["group_execution_results"] = []
    save_agent_result(final_output)

    for group in groups:
        group_id = group["group_id"]
        dependencies = group.get("depends_on", [])
        failed_dependencies = [
            dep for dep in dependencies
            if results_by_group.get(dep, {}).get("status") != "succeeded"
        ]

        if failed_dependencies:
            outcome = {
                "group_id": group_id,
                "status": "skipped_dependency_failed",
                "script_path": None,
                "stdout": "",
                "stderr": f"Required groups did not succeed: {failed_dependencies}",
                "returncode": None
            }
            print(f"\n[GROUP {group_id} SKIPPED] Failed dependencies: {failed_dependencies}")
        else:
            outcome = execute_group_program(
                user_request,
                analysis,
                optimization,
                group,
                results_by_group,
                final_output,
                failure_injection=failure_injection
            )

        results_by_group[group_id] = outcome
        final_output["group_execution_results"].append(outcome)
        save_agent_result(final_output)

        for operation_id in group["operation_ids"]:
            operation_status[operation_id] = outcome["status"]

        print(f"[GROUP {group_id} STATUS] {outcome['status']}")

    statuses = [item["status"] for item in results_by_group.values()]
    if statuses and all(status == "succeeded" for status in statuses):
        final_output["execution_status"] = "succeeded"
    elif any(status == "succeeded" for status in statuses):
        final_output["execution_status"] = "partially_succeeded"
    else:
        final_output["execution_status"] = "failed"

    final_output["operation_execution_status"] = operation_status
    save_agent_result(final_output)
    return final_output


# ============================================================
# 7. MAIN APPLICATION - STEPS 1, 2 AND 3
# ============================================================

def main():
    print("\nQwen 2.5 Coder 7B - Adaptive Execution Engine")
    print("Step 1: Requirement Analysis and Operation Planning")
    print("Step 2: Operation Assessment, Classification and Optimization")
    print("Step 3: Group Code Generation, Validation and Execution")
    print("Step 4: Failure Diagnosis, Safe Recovery and Re-execution")
    print("Controlled failure tests: missing import, partial side effect, dependency failure")
    print("Generated code is shown and requires your approval before execution.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        try:
            print("\n[STEP 1] Analyzing requirements and identifying operations...")
            analysis = analyze_requirements(user_input)
            print("\n[REQUIREMENT ANALYSIS]")
            print(json.dumps(analysis, indent=4, ensure_ascii=False))

            print("\n[STEP 2] Assessing operations and optimizing execution plan...")
            optimization = optimize_execution_plan(user_input, analysis)
            print("\n[OPERATION ASSESSMENTS]")
            print(json.dumps(optimization["operation_assessments"], indent=4, ensure_ascii=False))
            print("\n[SELECTED EXECUTION PLAN]")
            print("Selected strategy:", optimization["selected_strategy_id"])
            print("Reason:", optimization["selection_reason"])
            print(json.dumps(optimization["execution_groups"], indent=4, ensure_ascii=False))

            failure_injection = select_failure_injection(
                optimization["execution_groups"], analysis
            )

            final_output = {
                "original_user_request": user_input,
                "user_intent": analysis.get("user_intent"),
                "user_requirements": analysis["user_requirements"],
                "operations": analysis["operations"],
                "constraints": analysis.get("constraints", []),
                "assumptions": analysis.get("assumptions", []),
                "unresolved_inputs": analysis.get("unresolved_inputs", []),
                "overall_expected_result": analysis.get("overall_expected_result"),
                "operation_assessments": optimization["operation_assessments"],
                "candidate_strategies": optimization["candidate_strategies"],
                "selected_strategy_id": optimization["selected_strategy_id"],
                "selection_reason": optimization["selection_reason"],
                "optimization_summary": optimization["optimization_summary"],
                "execution_groups": optimization["execution_groups"],
                "execution_status": "not_started",
                "failure_injection_test": failure_injection,
                "group_execution_results": []
            }
            save_agent_result(final_output)

            print("\n[STEP 3] Ready to generate and execute group programs.")
            print("Review each generated program carefully before approving it.")
            final_output = execute_optimized_plan(
                user_input, analysis, optimization, final_output,
                failure_injection=failure_injection
            )

            print("\n" + "=" * 72)
            print("STEP 1, STEP 2, STEP 3 AND STEP 4 COMPLETED")
            print("Final execution status:", final_output["execution_status"])
            print("Results saved to agent_result.json")
            print("Generated programs saved in generated_scripts/")
            print("Task files should be created under workspaces/ when the generated code follows the workspace contract.")
            print("=" * 72)

        except Exception as ex:
            print(f"\n[WORKFLOW ERROR] {type(ex).__name__}: {ex}")
            print("Check agent_result.json for the last saved checkpoint, if available.")


if __name__ == "__main__":
    main()
