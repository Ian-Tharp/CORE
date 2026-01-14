"""Quick test endpoint for CORE without LLM calls"""
from fastapi import APIRouter
from app.models.core_state import COREState, UserIntent, ExecutionPlan, PlanStep, StepResult, EvaluationResult

router = APIRouter(prefix="/test", tags=["test"])

@router.get("/core-state")
async def test_core_state():
    """Test that CORE state model works"""
    state = COREState(user_input="Test input")

    # Add intent
    state.intent = UserIntent(
        type="task",
        description="Test task",
        confidence=0.9,
        requires_tools=True,
        tools_needed=["file_operations"]
    )

    # Add plan
    state.plan = ExecutionPlan(
        goal="Test goal",
        steps=[
            PlanStep(
                name="Step 1",
                description="Do something",
                tool="file_operations"
            )
        ]
    )

    # Add result
    state.step_results = [
        StepResult(
            step_id=state.plan.steps[0].id,
            status="success",
            outputs={"result": "test"}
        )
    ]

    # Add evaluation
    state.eval_result = EvaluationResult(
        overall_status="success",
        confidence=0.9,
        meets_requirements=True,
        quality_score=0.85,
        feedback="Test passed",
        next_action="finalize"
    )

    state.response = "Test completed successfully"

    return {
        "status": "ok",
        "run_id": state.run_id,
        "intent": state.intent.dict() if state.intent else None,
        "plan_steps": len(state.plan.steps) if state.plan else 0,
        "response": state.response
    }
