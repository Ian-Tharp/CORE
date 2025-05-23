from pydantic import BaseModel


class OrchestrationOutput(BaseModel):
    overall_plan: str
    task_plan: str
