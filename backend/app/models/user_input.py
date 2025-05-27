from pydantic import BaseModel


class UserInput(BaseModel):
    message_id: str
    user_input: str
