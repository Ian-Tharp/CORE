from fastapi import APIRouter, Request
from models.user_input import UserInput

router = APIRouter(prefix="/core")


@router.post("")
async def core_entry(user_input: UserInput, request: Request):
    # This is the entry point of the C.O.R.E cognitive engine.
    # This goes through each of the steps for the CORE flow, starting with:
    # Comprehension:
    #  - First take the user input/query and check if it is a command or a query.
    #  - If it is a command, then we route to the appropriate command handler.
    #  - Otherwise, if it is a query, we then need to process and go to the Comprehension node.
    #  - This will check the user's intent and check against the system's knowledge base and list of capabilities to see if we can/should process the user query.
    #  - If we can process the query, via the knowledge base and list of capabilities, then we route to the appropriate node.
    #  - If we cannot process the query, then we route to the conversation node.
    # Orchestration:
    #  - This will take the output of the Comprehension node and develop a plan, or course of action, to complete the user's request.
    #  - This will be based on the information from the Comprehension node and the system's knowledge base and list of capabilities.
    #  - This will generate a step by step plan to execute the user's request, and pass it along to the Reasoning node to be executed.
    # Reasoning:
    #  - This will take the plan from the Orchestration node and execute the steps in the plan.
    #  - This will be based on the information from the Orchestration node and the system's knowledge base and list of capabilities.
    # Evaluation:
    #  - Depending on the result of the reasoning step, either as a iteration or completion of the task/plan
    #    this will either go back to the Orchestration step to revise the plan or step if the result was Unsatisfactory.
    #  - If the result was Satisfactory, then we route to the Conversation step to complete the plan.
    # Conversation:
    #  - This is the node to send the final response to the user.
    pass


@router.post("/comprehension")
async def comprehension(user_input: UserInput, request: Request):
    pass


@router.post("/orchestration")
async def orchestration(user_input: UserInput, request: Request):
    pass


@router.post("/reasoning")
async def reasoning(user_input: UserInput, request: Request):
    pass


@router.post("/evaluation")
async def evaluation(user_input: UserInput, request: Request):
    pass
