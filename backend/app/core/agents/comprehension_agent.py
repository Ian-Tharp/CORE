"""
Comprehension class to handle the initial user input and validate relevancy of the system capability to answer the query.
"""

from app.dependencies import get_openai_client
from app.models.comprehension_output import ComprehensionOutput
from app.models.user_input import UserInput
from app.prompts.comprehension_agent_prompt import comprehension_agent_prompt


class ComprehensionAgent:
    def __init__(self, model: str):
        self.model = model or "gpt-4.1-mini"
        self.system_capabilities = comprehension_agent_prompt

    def run(self, user_input: UserInput) -> ComprehensionOutput:
        client = get_openai_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_capabilities},
                {"role": "user", "content": user_input},
            ],
        )
        if response.choices[0].message.content == "Within Expertise":
            return ComprehensionOutput.WITHIN_EXPERTISE
        else:
            return ComprehensionOutput.OUTSIDE_EXPERTISE

    def check_relevancy(user_input: UserInput) -> str:
        pass

    def check_knowledge_base(user_input: UserInput) -> str:
        pass
