from pydantic import BaseModel
from enum import Enum


class ReasoningTaskStatus(Enum):
    CREATED = ("Created",)
    AWAITING_APPROVAL = ("Awaiting Approval",)
    ACTIVE = ("Active",)
    AWAITING_HUMAN_INTERVENTION = ("Awaiting HITL",)
    PENDING = ("Pending",)
    ERROR = "Error"


class ReasoningOutput(BaseModel):
    task_id: str
    task_status: ReasoningTaskStatus
