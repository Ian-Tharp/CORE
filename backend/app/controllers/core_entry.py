from fastapi import APIRouter, Request
from models.user_input import UserInput

router = APIRouter(prefix="/core")


@router.post("")
async def core_entry(user_input: UserInput, request: Request):
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
