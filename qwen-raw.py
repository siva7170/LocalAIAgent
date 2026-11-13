import os
import json
from collections import defaultdict, deque
from llama_cpp import Llama


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
# 6. MAIN APPLICATION - STEPS 1 AND 2 ONLY
# ============================================================

def main():
    print("\nQwen 2.5 Coder 7B - Adaptive Plan Analyzer")
    print("Step 1: Requirement Analysis and Operation Planning")
    print("Step 2: Operation Assessment, Classification and Optimization")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        try:
            # STEP 1
            print("\n[STEP 1] Analyzing requirements and identifying operations...")

            analysis = analyze_requirements(user_input)

            print("\n[REQUIREMENT ANALYSIS]")
            print(json.dumps(analysis, indent=4, ensure_ascii=False))

            # STEP 2
            print("\n[STEP 2] Assessing operations and optimizing execution plan...")

            optimization = optimize_execution_plan(
                user_input,
                analysis
            )

            print("\n[OPERATION ASSESSMENTS]")
            print(json.dumps(
                optimization["operation_assessments"],
                indent=4,
                ensure_ascii=False
            ))

            print("\n[CANDIDATE STRATEGIES]")
            print(json.dumps(
                optimization["candidate_strategies"],
                indent=4,
                ensure_ascii=False
            ))

            print("\n[SELECTED EXECUTION PLAN]")
            print("Selected strategy:", optimization["selected_strategy_id"])
            print("Reason:", optimization["selection_reason"])
            print(json.dumps(
                optimization["execution_groups"],
                indent=4,
                ensure_ascii=False
            ))

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
                "execution_status": "not_started"
            }

            with open("agent_result.json", "w", encoding="utf-8") as file:
                json.dump(
                    final_output,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print("\n" + "=" * 65)
            print("STEP 1 AND STEP 2 COMPLETED")
            print("=" * 65)
            print("Execution has NOT started. No generated Python was executed.")
            print("Result saved to agent_result.json")

        except Exception as ex:
            print(f"\n[PLANNING ERROR] {type(ex).__name__}: {ex}")
            print("No execution was started.")


if __name__ == "__main__":
    main()
