from pydantic import BaseModel
from typing import Optional, List


class UserInput(BaseModel):
    message_id: str
    user_input: str
    model: Optional[str] = None
    # Prior-step context (optional, passed from the UI playground)
    comprehension_text: Optional[str] = None
    comprehension_route: Optional[str] = None
    orchestration_text: Optional[str] = None
    orchestration_plan: Optional[List[str]] = None
    reasoning_text: Optional[str] = None
