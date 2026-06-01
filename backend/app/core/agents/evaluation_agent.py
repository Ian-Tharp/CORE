"""
Evaluation Agent - Assesses execution quality and determines next action.

Responsibilities:
- Evaluate if the goal was accomplished
- Assess quality and confidence of outputs
- Decide next action: finalize, retry, or revise plan
"""

import json
import logging
from typing import Dict, List, Optional

from app.dependencies import get_openai_client_sync
from app.models.core_state import (
    UserIntent,
    ExecutionPlan,
    StepResult,
    EvaluationResult,
)
from app.utils.json_repair import safe_json_loads, extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-type rubrics
# ---------------------------------------------------------------------------

# Each rubric is a compact list of evaluation criteria specific to that intent
# type.  The rubric is injected into the LLM prompt so assessment is grounded
# in the right quality dimensions for the task at hand.
_RUBRICS: Dict[str, str] = {
    "code": (
        "Code-specific rubric:\n"
        "- Does the output compile / run without syntax errors?\n"
        "- Are edge cases and error conditions handled?\n"
        "- Is the implementation idiomatic and maintainable?\n"
        "- Are tests or documentation included if requested?"
    ),
    "research": (
        "Research-specific rubric:\n"
        "- Are claims backed by sources or reasoning?\n"
        "- Is the scope of the question adequately covered?\n"
        "- Are conflicting viewpoints acknowledged?\n"
        "- Is the depth appropriate for the stated goal?"
    ),
    "file_management": (
        "File management rubric:\n"
        "- Were all target files created / modified / deleted as requested?\n"
        "- Is the directory structure correct?\n"
        "- Are file permissions / encodings appropriate?\n"
        "- Was any existing data preserved where required?"
    ),
    "data_analysis": (
        "Data analysis rubric:\n"
        "- Is the analysis methodology sound?\n"
        "- Are statistical or aggregation results clearly presented?\n"
        "- Are edge cases (nulls, outliers) handled?\n"
        "- Is the output format appropriate for the audience?"
    ),
    "communication": (
        "Communication rubric:\n"
        "- Is the message tone appropriate for the context?\n"
        "- Is all required information included?\n"
        "- Is the message concise and free of ambiguity?\n"
        "- Were the correct recipients / channels used?"
    ),
    "task": (
        "General task rubric:\n"
        "- Was the stated goal fully accomplished?\n"
        "- Is the output complete and usable as delivered?\n"
        "- Were any explicit constraints respected?\n"
        "- Is the result consistent with the user's original request?"
    ),
    "conversation": (
        "Conversation rubric:\n"
        "- Was the user's question directly addressed?\n"
        "- Is the response accurate and well-reasoned?\n"
        "- Is the tone and length appropriate?\n"
        "- Were follow-up opportunities identified where relevant?"
    ),
}

# Fallback rubric for intent types not in the catalogue
_DEFAULT_RUBRIC = (
    "General evaluation rubric:\n"
    "- Was the stated goal accomplished?\n"
    "- Is the output complete and usable?\n"
    "- Were any constraints respected?\n"
    "- Is the result consistent with the user's request?"
)


def get_rubric_for_intent(intent_type: Optional[str]) -> str:
    """
    Return the evaluation rubric for the given intent type / category string.

    Normalises the type string (lower-case, strip, hyphen→underscore) and
    falls back to _DEFAULT_RUBRIC if no specific rubric is registered.

    Pass a ``UserIntent.task_category`` value (e.g. "code", "research") for
    task-specific rubrics, or a ``UserIntent.type`` string for broader ones.
    """
    if not intent_type:
        return _DEFAULT_RUBRIC
    normalised = intent_type.lower().strip().replace("-", "_").replace(" ", "_")
    return _RUBRICS.get(normalised, _DEFAULT_RUBRIC)


def get_rubric_for_user_intent(intent: Optional["UserIntent"]) -> str:  # noqa: F821
    """
    Return the evaluation rubric for a full UserIntent object.

    Prefers ``intent.task_category`` (fine-grained) over ``intent.type``
    (routing-level), falling back to _DEFAULT_RUBRIC if neither matches.
    """
    if intent is None:
        return _DEFAULT_RUBRIC
    # Prefer fine-grained task_category if present
    if intent.task_category:
        rubric = get_rubric_for_intent(intent.task_category)
        if rubric != _DEFAULT_RUBRIC:
            return rubric
    return get_rubric_for_intent(intent.type)


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
        step_results: List[StepResult],
    ) -> EvaluationResult:
        """
        Evaluate execution results using LLM for nuanced quality assessment.

        Calls the LLM to determine whether the user's request was satisfied,
        assess output quality, and decide the next action. Falls back to
        rule-based evaluation if the LLM call fails.

        Args:
            user_input: Original user input
            intent: The comprehension result
            plan: The execution plan that was run
            step_results: Results from each step execution

        Returns:
            EvaluationResult with LLM-assessed quality and next action
        """
        return self._llm_evaluate(user_input, plan, step_results, intent=intent)

    def _rule_based_evaluate(self, step_results: List[StepResult]) -> EvaluationResult:
        """
        Rule-based fallback evaluation when LLM is unavailable.

        Uses step success/failure counts to determine outcome.
        Called automatically when _llm_evaluate raises an exception.
        """
        if not step_results:
            return EvaluationResult(
                overall_status="failure",
                confidence=0.4,
                meets_requirements=False,
                quality_score=0.5,
                feedback="No steps were executed. User input needed.",
                next_action="ask_user",
            )

        failed_steps = [r for r in step_results if r.status == "failure"]
        all_succeeded = not failed_steps
        success_rate = (
            sum(1 for r in step_results if r.status == "success") / len(step_results)
            if step_results
            else 0.0
        )

        if all_succeeded:
            return EvaluationResult(
                overall_status="success",
                confidence=0.9,
                meets_requirements=True,
                quality_score=0.85,
                feedback=f"All {len(step_results)} steps completed successfully.",
                next_action="finalize",
            )
        elif failed_steps and len(failed_steps) < len(step_results):
            return EvaluationResult(
                overall_status="needs_retry",
                confidence=0.6,
                meets_requirements=False,
                quality_score=success_rate,
                feedback=f"{len(failed_steps)} step(s) failed. Retrying failed steps.",
                next_action="retry_step",
                retry_step_id=failed_steps[0].step_id,
            )
        elif step_results and len(failed_steps) == len(step_results):
            return EvaluationResult(
                overall_status="needs_revision",
                confidence=0.3,
                meets_requirements=False,
                quality_score=0.2,
                feedback="All steps failed. The plan may be flawed and needs revision.",
                next_action="revise_plan",
            )
        else:
            return EvaluationResult(
                overall_status="failure",
                confidence=0.4,
                meets_requirements=False,
                quality_score=0.5,
                feedback="Unable to determine outcome. User input needed.",
                next_action="ask_user",
            )

    def _llm_evaluate(
        self,
        user_input: str,
        plan: ExecutionPlan,
        step_results: List[StepResult],
        intent: Optional[UserIntent] = None,
    ) -> EvaluationResult:
        """
        Use LLM to evaluate execution quality with nuanced assessment.

        Builds a rich context from the user request, intent, plan goal, and
        per-step results, then asks the LLM to decide:
        - Did the execution satisfy the user's request?
        - What is the output quality?
        - What should happen next (finalize / retry / revise / ask)?

        Falls back to rule-based evaluation if the LLM call fails.

        Args:
            user_input: Original user request text
            plan: The execution plan that was run
            step_results: Results for each executed step
            intent: Optional comprehension result providing task context
        """
        # Build intent context line
        if intent:
            intent_line = f"Intent type: {intent.type} — {intent.description}"
        else:
            intent_line = "Intent: not available"

        # Build per-step summaries, including output snippets
        step_lines = []
        for i, result in enumerate(step_results, 1):
            output_preview = str(result.outputs.get("result", result.outputs))[:200]
            error_suffix = f" [error: {result.error}]" if result.error else ""
            step_lines.append(
                f"{i}. [{result.status.upper()}] {output_preview}{error_suffix}"
            )

        steps_block = "\n".join(step_lines) if step_lines else "No steps were executed."

        # Rubric: inject task-type-specific evaluation criteria
        rubric = get_rubric_for_user_intent(intent)

        context = f"""User requested: {user_input}

{intent_line}
Plan goal: {plan.goal}
Plan steps: {len(plan.steps)}

Step execution results:
{steps_block}

{rubric}

Evaluate whether the user's request was satisfied by the execution above."""

        try:
            client = get_openai_client_sync()
            logger.info("Evaluation assessing execution with model=%s", self.model)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.2,  # Low temperature for consistent evaluation
            )

            content = response.choices[0].message.content
            logger.info("Evaluation LLM response: %s", content)
            if not content:
                raise ValueError("Empty response from LLM")

            # Extract and repair JSON from response
            extracted = extract_json_object(content)
            if extracted:
                logger.debug("Evaluation: extracted JSON object from response")
            else:
                extracted = content

            data = safe_json_loads(extracted)
            if data is None:
                raise ValueError(
                    f"Could not parse JSON from response: {content[:200]}..."
                )

            # Validate next_action — default to finalize if LLM returns garbage
            valid_actions = {"finalize", "retry_step", "revise_plan", "ask_user"}
            next_action = data.get("next_action", "finalize")
            if next_action not in valid_actions:
                logger.warning(
                    "LLM returned invalid next_action '%s', defaulting to 'finalize'",
                    next_action,
                )
                next_action = "finalize"

            # Validate overall_status
            valid_statuses = {"success", "failure", "needs_revision", "needs_retry"}
            overall_status = data.get("overall_status", "success")
            if overall_status not in valid_statuses:
                overall_status = "success"

            return EvaluationResult(
                overall_status=overall_status,
                confidence=float(data.get("confidence", 0.7)),
                meets_requirements=bool(data.get("meets_requirements", True)),
                quality_score=float(data.get("quality_score", 0.7)),
                feedback=str(data.get("feedback", "")),
                next_action=next_action,
                retry_step_id=data.get("retry_step_id"),
            )

        except Exception as exc:
            logger.warning(
                "LLM evaluation failed for model=%s (using rule-based fallback): %s",
                self.model,
                exc,
            )
            return self._rule_based_evaluate(step_results)
