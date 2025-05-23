"""
Reasoning class to handle the actual processing of the plan from the orchestration class. This is where we can rely
on the Knowledge Base in order to complete the workflow generated from the orchestration. Or utilize custom logic
to complete the workflow.
"""

from app.models.user_input import UserInput


def execute_plan(user_input: UserInput, plan: str) -> str:
    pass
