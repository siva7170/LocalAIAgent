
import os
import json
from llama_cpp import Llama


# ============================================================
# 1. MODEL INITIALIZATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
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

def ask_llm(system_prompt, user_prompt, max_tokens=2048):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1,
        top_p=0.8,
        repeat_penalty=1.1,
        max_tokens=max_tokens,
        response_format={
            "type": "json_object"
        }
    )

    content = response["choices"][0]["message"]["content"]

    try:
        return json.loads(content)

    except json.JSONDecodeError as ex:
        raise ValueError(
            f"Invalid JSON returned by model: {ex}\n{content}"
        )


# ============================================================
# 3. STAGE 1 - REQUIREMENT ANALYSIS AND PLAN GENERATION
# ============================================================

def analyze_requirements(user_input):

    system_prompt = """
You are an expert software developer,
requirements analyst and project planner.

YOUR RESPONSIBILITIES:

1. Understand the user's actual intention.
2. Identify all user requirements.
3. Break the requirements into logical,
   ordered and independent plans.
4. Each plan must represent one achievable task.
5. Plans must be arranged in execution order.
6. Identify dependencies between plans.

IMPORTANT:

- Do not generate source code.
- Do not generate Python programs.
- Do not execute any operation.
- Generate plans only.
- Return valid JSON only.

REQUIRED JSON FORMAT:

{
    "user_requirements": [
        "Requirement 1",
        "Requirement 2"
    ],
    "plans": [
        {
            "plan_id": 1,
            "description": "Description of the plan",
            "objective": "Expected objective",
            "depends_on": []
        }
    ]
}

The plan_id must start from 1 and increment
sequentially.

Ensure that all plans together satisfy
the user's requirements.
"""

    result = ask_llm(
        system_prompt,
        user_input,
        max_tokens=2048
    )

    if not isinstance(result.get("plans"), list):
        raise ValueError("Model did not return a valid plans array.")

    for index, plan in enumerate(result["plans"], start=1):
        plan["plan_id"] = index

    return result


# ============================================================
# 4. STAGE 2 - DYNAMIC SYSTEM PROMPT GENERATOR
# ============================================================

def create_plan_system_prompt(
    plan_number,
    total_plans,
    previous_results
):

    if plan_number == 1:

        system_prompt = """
You are an expert Python developer and plan assessor.

THIS IS THE FIRST PLAN.

YOUR RESPONSIBILITIES:

1. Assess the current plan.
2. Understand its objective.
3. Identify the requirements to achieve it.
4. Determine the appropriate implementation.
5. Generate a complete Python program to achieve
   the current plan.
6. Identify assumptions and possible errors.

IMPORTANT:

- This is the first plan.
- There are no previous plan results.
- Do not assume that previous operations were performed.
- Generate Python code only for the current plan.
- Do not execute the generated Python program.
- Return valid JSON only.

PYTHON PROGRAM OUTPUT RULES:

1. The python_program field must contain
   raw executable Python source code only.

2. Never include Markdown code fences.

3. Never include explanations inside the
   python_program field.

4. Ensure the generated program is syntactically valid.

REQUIRED JSON FORMAT:

{
    "plan_id": 1,
    "assessment": "Assessment of the current plan",
    "implementation_approach": "How to achieve this plan",
    "python_program": "Complete Python source code",
    "expected_result": "Expected result",
    "execution_status": "not_executed"
}
"""

    else:

        system_prompt = f"""
You are an expert Python developer,
plan assessor and implementation analyst.

THIS IS PLAN NUMBER {plan_number}
OUT OF {total_plans}.

YOUR RESPONSIBILITIES:

1. Assess the current plan.
2. Review the outputs and results of all
   previously processed plans.
3. Identify dependencies on previous plans.
4. Determine whether previous results can be reused.
5. Identify missing information or incomplete results.
6. Develop the implementation for the current plan.
7. Generate a complete Python program to achieve
   the current plan.
8. Avoid unnecessarily repeating previous work.

IMPORTANT:

- This is NOT the first plan.
- Previous plan results are provided as context.
- Analyze the previous results before generating code.
- Use relevant outputs from previous plans.
- Do not assume that generated programs were executed.
- Do not claim that previous operations succeeded
  unless their execution results confirm success.
- Generate Python code only for the current plan.
- Do not execute the generated Python program.
- Return valid JSON only.

PYTHON PROGRAM OUTPUT RULES:

1. The python_program field must contain
   raw executable Python source code only.

2. Never include Markdown code fences.

3. Never include explanations inside the
   python_program field.

4. Ensure the generated program is syntactically valid.

REQUIRED JSON FORMAT:

{{
    "plan_id": {plan_number},
    "assessment": "Assessment of current plan",
    "previous_plan_analysis": "Analysis of previous results",
    "dependencies": "Required previous plan outputs",
    "implementation_approach": "How to achieve this plan",
    "python_program": "Complete Python source code",
    "expected_result": "Expected result",
    "execution_status": "not_executed"
}}
"""

    return system_prompt


# ============================================================
# 5. STAGE 3 - PROCESS EACH PLAN
# ============================================================

def process_plans(user_input, analysis):

    plans = analysis["plans"]

    total_plans = len(plans)

    previous_results = []

    final_results = []

    print("\nUSER REQUIREMENTS:")

    for requirement in analysis.get("user_requirements", []):
        print(f"- {requirement}")

    print(f"\nTOTAL PLANS: {total_plans}")

    for index, plan in enumerate(plans, start=1):

        plan_number = index

        print("\n" + "=" * 60)
        print(f"PROCESSING PLAN {plan_number}/{total_plans}")
        print("=" * 60)

        print("Plan:", plan["description"])

        # Generate dynamic system prompt
        system_prompt = create_plan_system_prompt(
            plan_number,
            total_plans,
            previous_results
        )

        # Prepare current plan context
        context = {
            "original_user_request": user_input,
            "all_user_requirements": analysis["user_requirements"],
            "all_plans": plans,
            "current_plan": plan,
            "previous_plan_results": previous_results
        }

        user_prompt = f"""
Assess and implement the current plan.

CURRENT CONTEXT:

{json.dumps(context, indent=2)}

Generate the required Python program.

Return JSON only.
"""

        # Ask LLM
        result = ask_llm(
            system_prompt,
            user_prompt,
            max_tokens=3072
        )

        # Validate response
        required_fields = [
            "assessment",
            "implementation_approach",
            "python_program",
            "expected_result"
        ]

        if plan_number > 1:
            required_fields.extend([
                "previous_plan_analysis",
                "dependencies"
            ])

        missing_fields = [
            field for field in required_fields
            if field not in result
        ]

        if missing_fields:
            raise ValueError(
                f"Plan {plan_number}: Missing JSON fields: "
                f"{missing_fields}"
            )

        result["plan_id"] = plan_number
        result["plan_description"] = plan["description"]
        result["execution_status"] = "not_executed"

        # Display result
        print("\nPLAN RESULT:")
        print(json.dumps(result, indent=4))

        # Store result for the next plan
        previous_results.append(result)

        final_results.append(result)

    return final_results


# ============================================================
# 6. MAIN APPLICATION
# ============================================================

def main():

    print("\nQwen 2.5 Coder 3B - Multi-Plan Agent")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        try:

            # STAGE 1
            print("\n[STAGE 1] Analyzing user requirements...")

            analysis = analyze_requirements(user_input)

            print("\nGENERATED PLANS:")
            print(json.dumps(analysis, indent=4))

            # STAGE 2 & 3
            print("\n[STAGE 2] Assessing and processing plans...")

            results = process_plans(
                user_input,
                analysis
            )

            # FINAL JSON
            final_output = {
                "user_requirements": analysis["user_requirements"],
                "total_plans": len(analysis["plans"]),
                "plan_results": results
            }

            print("\n" + "=" * 60)
            print("FINAL OUTPUT")
            print("=" * 60)

            print(json.dumps(
                final_output,
                indent=4
            ))

            # Save final output
            with open(
                "agent_result.json",
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    final_output,
                    file,
                    indent=4
                )

            print("\nResult saved to agent_result.json")

        except Exception as ex:

            print(f"\nERROR: {ex}")


if __name__ == "__main__":
    main()
