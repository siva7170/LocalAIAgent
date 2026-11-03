class AgentContext:

    def __init__(self, question):

        self.original_question = question
        self.execution_history = []
        self.completed_tasks = []
        self.failed_tasks = []

    def add_result(self, result, task):

        self.execution_history.append({
            **result.model_dump(),
            "arguments": task.arguments
        })

        if result.success:
            self.completed_tasks.append(result.task_id)
        else:
            self.failed_tasks.append(result.task_id)

    def get_history(self):
        return self.execution_history