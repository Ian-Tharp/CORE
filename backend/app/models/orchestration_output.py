from pydantic import BaseModel


class OrchestrationOutput(BaseModel):
    orchestration_id: str
    overall_plan: str
    task_plan: str
