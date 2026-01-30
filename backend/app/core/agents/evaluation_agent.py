"""
Evaluation Agent - Assesses execution quality and determines next action.

Responsibilities:
- Evaluate if the goal was accomplished
- Assess quality and confidence of outputs
- Decide next action: finalize, retry, or revise plan

RSI TODO: Implement LLM-based quality assessment
RSI TODO: Add rubric-based evaluation for different task types
RSI TODO: Integrate user feedback into evaluation criteria
"""

import json
import logging
from typing import List, Optional

from app.dependencies import get_openai_client_sync
from app.models.core_state import (
    UserIntent,
    ExecutionPlan,
    StepResult,
    EvaluationResult
)
from app.utils.json_repair import safe_json_loads, extract_json_object

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """
    Evaluation Agent - Fourth stage of CORE pipeline.

    Assesses execution outcomes and determines if we're done or need more work.
    """

    def __init__(self, model: str = "gpt-oss:20b"):
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for evaluation."""
        return """You are the Evaluation layer of the CORE cognitive system.

Your job is to assess whether the execution met the user's requirements.

Analyze:
1. Did all steps complete successfully?
2. Do the outputs satisfy the original goal?
3. Is the quality acceptable?
4. What's your confidence in the result?

Determine next action:
- **finalize**: Task completed successfully, return results to user
- **retry_step**: A specific step failed, retry it
- **revise_plan**: The plan itself was flawed, need to re-orchestrate
- **ask_user**: Need user input to proceed

Respond in JSON format:
{
  "overall_status": "success|failure|needs_revision|needs_retry",
  "confidence": 0.0-1.0,
  "meets_requirements": true|false,
  "quality_score": 0.0-1.0,
  "feedback": "Detailed evaluation feedback",
  "next_action": "finalize|retry_step|revise_plan|ask_user",
  "retry_step_id": "step_id_or_null"
}"""

    def evaluate_execution(
        self,
        user_input: str,
        intent: Optional[UserIntent],
        plan: ExecutionPlan,
        step_results: List[StepResult]
    ) -> EvaluationResult:
        """
        Evaluate the execution results.

        Args:
            user_input: Original user input
            intent: The comprehension result
            plan: The execution plan that was run
            step_results: Results from each step execution

        Returns:
            EvaluationResult with assessment and next action
        """
        # For now, use simple rule-based evaluation
        # RSI TODO: Replace with LLM-based evaluation for nuanced assessment

        # Check if all steps succeeded
        all_succeeded = all(r.status == "success" for r in step_results)
        failed_steps = [r for r in step_results if r.status == "failure"]

        # Calculate confidence based on step success rate
        if step_results:
            success_rate = sum(1 for r in step_results if r.status == "success") / len(step_results)
        else:
            success_rate = 0.0

        # Determine overall status and next action
        if all_succeeded:
            overall_status = "success"
            next_action = "finalize"
            meets_requirements = True
            quality_score = 0.85  # High quality for all successful
            feedback = f"All {len(step_results)} steps completed successfully."
            retry_step_id = None
            confidence = 0.9

        elif failed_steps and len(failed_steps) < len(step_results):
            # Some steps failed
            overall_status = "needs_retry"
            next_action = "retry_step"
            meets_requirements = False
            quality_score = success_rate
            feedback = f"{len(failed_steps)} step(s) failed. Retrying failed steps."
            retry_step_id = failed_steps[0].step_id  # Retry first failed step
            confidence = 0.6

        elif len(failed_steps) == len(step_results):
            # All steps failed - plan is bad
            overall_status = "needs_revision"
            next_action = "revise_plan"
            meets_requirements = False
            quality_score = 0.2
            feedback = "All steps failed. The plan may be flawed and needs revision."
            retry_step_id = None
            confidence = 0.3

        else:
            # Edge case
            overall_status = "failure"
            next_action = "ask_user"
            meets_requirements = False
            quality_score = 0.5
            feedback = "Unable to determine outcome. User input needed."
            retry_step_id = None
            confidence = 0.4

        return EvaluationResult(
            overall_status=overall_status,
            confidence=confidence,
            meets_requirements=meets_requirements,
            quality_score=quality_score,
            feedback=feedback,
            next_action=next_action,
            retry_step_id=retry_step_id
        )

    def _llm_evaluate(
        self,
        user_input: str,
        plan: ExecutionPlan,
        step_results: List[StepResult]
    ) -> EvaluationResult:
        """
        Use LLM to evaluate execution quality (future implementation).

        RSI TODO: Implement this when we have more sophisticated evaluation needs
        """
        client = get_openai_client_sync()

        # Build evaluation context
        context = f"""User wanted: {user_input}

Plan created: {plan.goal}
Steps: {len(plan.steps)}

Execution results:
"""
        for i, result in enumerate(step_results, 1):
            context += f"{i}. {result.status}: {result.outputs}\n"

        try:
            logger.info(f"Evaluation assessing execution with model={self.model}")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.2,  # Low temperature for consistent evaluation
                # Note: Ollama may not support response_format, removed for compatibility
            )

            content = response.choices[0].message.content
            logger.info(f"Evaluation LLM response: {content}")
            if not content:
                raise ValueError("Empty response from LLM")

            # Extract and repair JSON from response
            extracted = extract_json_object(content)
            if extracted:
                logger.info(f"Evaluation: Extracted JSON object")
            else:
                extracted = content
            
            data = safe_json_loads(extracted)
            if data is None:
                raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

            return EvaluationResult(
                overall_status=data.get("overall_status", "success"),
                confidence=float(data.get("confidence", 0.7)),
                meets_requirements=data.get("meets_requirements", True),
                quality_score=float(data.get("quality_score", 0.7)),
                feedback=data.get("feedback", ""),
                next_action=data.get("next_action", "finalize"),
                retry_step_id=data.get("retry_step_id")
            )

        except Exception as e:
            # Fallback to success on error
            return EvaluationResult(
                overall_status="success",
                confidence=0.5,
                meets_requirements=True,
                quality_score=0.5,
                feedback=f"Evaluation error: {str(e)}. Defaulting to success.",
                next_action="finalize"
            )
