
from ai.planner import create_plan
from ai.responder import generate_answer
from engine.executor import execute_plan

from config import MAX_ROUNDS


def process_question(question):

    history = []

    print("\n[AI] Understanding your request...")

    for round_number in range(1, MAX_ROUNDS + 1):

        print(f"\n[PLANNER] Planning round {round_number}...")

        plan = create_plan(question, history)

        print("\n[EXECUTION PLAN]")
        print(plan.get("understanding", ""))

        if not plan["tasks"]:

            if plan.get("done", False):
                print("\n[PLANNER] All tasks completed.")
                break

            print("\n[PLANNER] No executable tasks returned.")
            break

        print("\n[PLANNER] Tasks identified:")

        for task in plan["tasks"]:
            print(
                f"{task.get('id')}. "
                f"{task.get('purpose')} "
                f"-> {task.get('command')}"
            )

        # Execute tasks sequentially
        results = execute_plan(plan)

        history.extend(results)

        if plan.get("done", False):
            break

    print("\n[AI] Preparing final response...")

    answer = generate_answer(question, history)

    print("\n" + "=" * 60)
    print("AI RESPONSE")
    print("=" * 60)
    print(answer)
    print("=" * 60)


def main():

    print("====================================")
    print("       LOCAL AI ASSISTANT")
    print("====================================")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            process_question(question)

        except Exception as ex:
            print(f"\n[ERROR] {ex}")


if __name__ == "__main__":
    main()
