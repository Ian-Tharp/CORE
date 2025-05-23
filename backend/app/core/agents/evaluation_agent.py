"""
Evaluation class to process the output of the reasoning agent and determine the viability of the output, also
checking if the output meets a satisfactory criteria matching the plan from the orchestration agent.
"""

from app.models.user_input import UserInput


def evaluate_output(user_input: UserInput, reasoning_output: str) -> str:
    pass
