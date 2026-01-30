"""
Task Router Service

Core routing logic for the CORE Task Routing Engine.
Implements the Orchestration phase of the CORE loop by deciding which agent 
gets which task based on capabilities, load, performance history, and trust.

Architecture Context:
1. Comprehension analyzes what's needed
2. **Orchestration routes it** ← THIS IS WHAT THIS SERVICE IMPLEMENTS
3. Reasoning executes it
4. Evaluation checks the output
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from app.models.task_models import (
    Task, TaskAssignment, TaskResult, TaskRoutingScore, TaskStatus, 
    AgentResponse, TaskType, AgentTaskMetrics
)
from app.repository.task_repository import (
    create_task, update_task_status, get_queued_tasks, create_task_assignment,
    create_task_result, get_task_metrics, get_agent_task_metrics
)
from app.repository.instance_repository import (
    get_instances_with_trust_metrics, increment_task_completed, 
    increment_task_failed, increment_task_refused, record_override
)
from app.services.agent_registry import agent_registry

logger = logging.getLogger(__name__)


class TaskRouter:
    """
    Central task routing service for the CORE system.
    
    Implements multi-factor agent selection based on:
    - Capability matching
    - Current load
    - Trust scores from historical performance
    - Model preference alignment
    - Task type performance history
    """
    
    def __init__(self):
        self._routing_analytics: Dict[str, Any] = {}
        self._task_type_performance: Dict[Tuple[UUID, str], Dict[str, Any]] = {}
    
    async def route_task(self, task: Task) -> Optional[TaskAssignment]:
        """
        Route a task to the best available agent.
        
        Process:
        1. Filter agents by required capabilities
        2. Filter by availability (status == 'ready', load < 80%)
        3. Score remaining agents by multiple factors
        4. Assign to highest-scoring agent
        5. If no agents available, queue the task
        6. If agent refuses, re-route to next best
        
        Args:
            task: Task to be routed
            
        Returns:
            TaskAssignment if successfully routed, None if queued
        """
        logger.info(f"Routing task {task.id} (type: {task.task_type}, priority: {task.priority})")
        
        # Get available agents with trust metrics
        agents_with_metrics = await get_instances_with_trust_metrics()
        
        if not agents_with_metrics:
            logger.warning("No agents available for task routing")
            await update_task_status(task.id, TaskStatus.QUEUED)
            return None
        
        # Filter by capabilities and availability
        eligible_agents = await self._filter_eligible_agents(task, agents_with_metrics)
        
        if not eligible_agents:
            logger.info(f"No eligible agents for task {task.id}, queuing")
            await update_task_status(task.id, TaskStatus.QUEUED)
            return None
        
        # Score agents for this task
        agent_scores = await self._score_agents(task, eligible_agents)
        
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Try to assign to highest-scoring agent
        best_agent = agent_scores[0]
        
        logger.info(
            f"Selected agent {best_agent.agent_id} for task {task.id} "
            f"(score: {best_agent.total_score:.3f})"
        )
        
        # Create assignment
        assignment = TaskAssignment(
            task_id=task.id,
            agent_id=best_agent.agent_id,
            agent_response=AgentResponse.ACCEPT,  # Optimistic assignment
            confidence_score=best_agent.total_score
        )
        
        # Update task status
        await update_task_status(
            task.id, 
            TaskStatus.ASSIGNED, 
            assigned_agent_id=best_agent.agent_id
        )
        
        # Record assignment
        await create_task_assignment(assignment)
        
        # Notify agent via registry
        await self._notify_agent_assignment(best_agent.agent_id, task)
        
        return assignment
    
    async def _filter_eligible_agents(
        self, 
        task: Task, 
        agents_with_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter agents by capabilities and availability.
        
        Args:
            task: Task requiring agents
            agents_with_metrics: List of agent data with trust metrics
            
        Returns:
            List of eligible agents
        """
        eligible = []
        
        for agent_data in agents_with_metrics:
            instance = agent_data["instance"]
            trust_metrics = agent_data["trust_metrics"]
            
            # Check if agent is ready and healthy
            if instance.status.value != "ready":
                continue
                
            # Check if agent is in registry (actually online)
            if instance.agent_id not in agent_registry.get_healthy_agents():
                continue
            
            # Check capability requirements
            if task.required_capabilities:
                agent_capabilities = set(instance.capabilities or [])
                required_capabilities = set(task.required_capabilities)
                
                if not required_capabilities.issubset(agent_capabilities):
                    logger.debug(
                        f"Agent {instance.agent_id} missing capabilities: "
                        f"{required_capabilities - agent_capabilities}"
                    )
                    continue
            
            # Check current load (< 80% capacity)
            current_load = await self._get_agent_current_load(instance.id)
            max_load = 5  # Maximum concurrent tasks per agent
            
            if current_load >= int(max_load * 0.8):
                logger.debug(f"Agent {instance.agent_id} at capacity: {current_load}/{max_load}")
                continue
            
            # Add to eligible list
            eligible.append({
                "instance": instance,
                "trust_metrics": trust_metrics,
                "current_load": current_load,
                "max_load": max_load
            })
        
        logger.debug(f"Found {len(eligible)} eligible agents for task {task.id}")
        return eligible
    
    async def _score_agents(
        self, 
        task: Task, 
        eligible_agents: List[Dict[str, Any]]
    ) -> List[TaskRoutingScore]:
        """
        Score eligible agents for task assignment.
        
        Scoring factors:
        - Capability match score (exact match > partial)
        - Load score (lower load = higher score)
        - Trust score from historical performance
        - Model preference alignment
        - Historical performance on this task type
        
        Args:
            task: Task to score agents for
            eligible_agents: List of eligible agent data
            
        Returns:
            List of TaskRoutingScore objects
        """
        scores = []
        
        for agent_data in eligible_agents:
            instance = agent_data["instance"]
            trust_metrics = agent_data["trust_metrics"]
            current_load = agent_data["current_load"]
            max_load = agent_data["max_load"]
            
            # Calculate individual scores
            capability_score = self._calculate_capability_match_score(task, instance)
            load_score = self._calculate_load_score(current_load, max_load)
            trust_score = trust_metrics.get("trust_score", 0.5)
            model_score = self._calculate_model_preference_score(task, instance)
            task_type_score = await self._calculate_task_type_performance_score(
                task, instance.id
            )
            
            # Combined score with weights
            weights = {
                "capability": 0.35,
                "load": 0.20,
                "trust": 0.25,
                "model": 0.10,
                "task_type": 0.10
            }
            
            total_score = (
                capability_score * weights["capability"] +
                load_score * weights["load"] +
                trust_score * weights["trust"] +
                model_score * weights["model"] +
                task_type_score * weights["task_type"]
            )
            
            routing_score = TaskRoutingScore(
                agent_id=instance.id,
                total_score=total_score,
                capability_match_score=capability_score,
                load_score=load_score,
                trust_score=trust_score,
                model_preference_score=model_score,
                task_type_performance_score=task_type_score
            )
            
            scores.append(routing_score)
            
            logger.debug(
                f"Agent {instance.agent_id} scored {total_score:.3f} "
                f"(cap: {capability_score:.2f}, load: {load_score:.2f}, "
                f"trust: {trust_score:.2f}, model: {model_score:.2f}, "
                f"task_type: {task_type_score:.2f})"
            )
        
        return scores
    
    def _calculate_capability_match_score(self, task: Task, instance) -> float:
        """Calculate how well agent capabilities match task requirements."""
        if not task.required_capabilities:
            return 1.0  # No requirements means any agent can do it
        
        agent_capabilities = set(instance.capabilities or [])
        required_capabilities = set(task.required_capabilities)
        
        if not required_capabilities:
            return 1.0
        
        # Exact match bonus
        if required_capabilities.issubset(agent_capabilities):
            # More capabilities = slightly higher score
            extra_capabilities = len(agent_capabilities) - len(required_capabilities)
            bonus = min(0.1, extra_capabilities * 0.02)  # Cap bonus at 10%
            return 1.0 + bonus
        
        # Partial match (shouldn't happen due to filtering, but safety check)
        match_count = len(required_capabilities.intersection(agent_capabilities))
        return match_count / len(required_capabilities)
    
    def _calculate_load_score(self, current_load: int, max_load: int) -> float:
        """Calculate load score (lower load = higher score)."""
        if max_load == 0:
            return 0.0
        
        load_ratio = current_load / max_load
        return max(0.0, 1.0 - load_ratio)
    
    def _calculate_model_preference_score(self, task: Task, instance) -> float:
        """Calculate model preference alignment score."""
        if not task.preferred_model:
            return 1.0  # No preference means no penalty
        
        # For now, use a simple heuristic
        # In real implementation, this would check agent's configured model
        
        # Default model preferences by role
        role_model_preferences = {
            "researcher": ["ollama/llama3.2", "claude-sonnet"],
            "analyst": ["claude-sonnet", "gpt-4"],
            "writer": ["claude-sonnet", "gpt-4"],
            "coder": ["claude-sonnet", "gpt-4-turbo"],
            "monitor": ["ollama/llama3.2"]
        }
        
        preferred_models = role_model_preferences.get(instance.agent_role, [])
        
        if task.preferred_model in preferred_models:
            return 1.0
        elif preferred_models:  # Has preferences but doesn't match
            return 0.7
        else:  # No preferences defined
            return 0.85
    
    async def _calculate_task_type_performance_score(
        self, 
        task: Task, 
        agent_id: UUID
    ) -> float:
        """Calculate agent's historical performance on this task type."""
        cache_key = (agent_id, task.task_type)
        
        # Check cache first
        if cache_key in self._task_type_performance:
            cached = self._task_type_performance[cache_key]
            # Refresh cache if older than 1 hour
            if datetime.utcnow() - cached["timestamp"] < timedelta(hours=1):
                return cached["score"]
        
        # Calculate from metrics
        metrics = await get_agent_task_metrics()
        agent_metrics = next((m for m in metrics if m.agent_id == agent_id), None)
        
        if not agent_metrics or agent_metrics.total_assigned == 0:
            # No history - use neutral score
            score = 0.5
        else:
            # Weight success rate and completion time
            success_rate = agent_metrics.success_rate
            
            # Bonus for fast completion (if we have duration data)
            time_bonus = 0.0
            if agent_metrics.avg_duration_ms:
                # This would be better with task-type-specific averages
                # For now, use a simple heuristic
                avg_minutes = agent_metrics.avg_duration_ms / (1000 * 60)
                if avg_minutes < 5:  # Fast
                    time_bonus = 0.1
                elif avg_minutes > 30:  # Slow
                    time_bonus = -0.1
            
            score = min(1.0, success_rate + time_bonus)
        
        # Cache result
        self._task_type_performance[cache_key] = {
            "score": score,
            "timestamp": datetime.utcnow()
        }
        
        return score
    
    async def _get_agent_current_load(self, agent_id: UUID) -> int:
        """Get agent's current task load."""
        metrics = await get_agent_task_metrics()
        agent_metrics = next((m for m in metrics if m.agent_id == agent_id), None)
        return agent_metrics.current_load if agent_metrics else 0
    
    async def _notify_agent_assignment(self, agent_id: UUID, task: Task) -> None:
        """Notify agent of task assignment via registry."""
        # This would integrate with the agent registry to push task to agent
        # For now, just log the assignment
        logger.info(f"Notified agent {agent_id} of task assignment: {task.id}")
        
        # TODO: Integrate with agent_registry.assign_task() when available
    
    async def handle_task_refusal(
        self, 
        task_id: UUID, 
        agent_id: UUID, 
        reason: str, 
        suggested_agent: Optional[str] = None
    ) -> Optional[TaskAssignment]:
        """
        Handle task refusal from an agent.
        
        Process:
        1. Record refusal in trust metrics
        2. Re-route to suggested agent or next best
        3. If all agents refuse, escalate to human
        
        Args:
            task_id: ID of refused task
            agent_id: ID of refusing agent
            reason: Reason for refusal
            suggested_agent: Optional alternative agent suggestion
            
        Returns:
            New TaskAssignment if re-routed, None if escalated
        """
        logger.info(f"Agent {agent_id} refused task {task_id}: {reason}")
        
        # Record refusal
        refusal_assignment = TaskAssignment(
            task_id=task_id,
            agent_id=agent_id,
            agent_response=AgentResponse.REFUSE,
            refusal_reason=reason,
            suggested_agent=suggested_agent,
            confidence_score=0.0
        )
        
        await create_task_assignment(refusal_assignment)
        
        # Update trust metrics
        # Get instance ID from agent_id
        instances = await get_instances_with_trust_metrics()
        instance = next((inst for inst in instances if inst["instance"].id == agent_id), None)
        
        if instance:
            await increment_task_refused(instance["instance"].id)
        
        # Try to re-route
        # TODO: Implement re-routing logic
        # For now, just mark as failed
        await update_task_status(task_id, TaskStatus.FAILED, result={"error": "Agent refused task"})
        
        logger.warning(f"Task {task_id} marked as failed due to refusal - re-routing not implemented")
        
        return None
    
    async def handle_task_completion(
        self, 
        task_id: UUID, 
        result: Dict[str, Any], 
        duration_ms: int,
        agent_id: UUID,
        model_used: str,
        tokens_used: Optional[int] = None
    ) -> None:
        """
        Handle successful task completion.
        
        Process:
        1. Update task status
        2. Update trust metrics (tasks_completed++, avg_duration)
        3. Record task result
        
        Args:
            task_id: Completed task ID
            result: Task result data
            duration_ms: Task execution duration
            agent_id: Agent that completed the task
            model_used: LLM model used
            tokens_used: Optional token count
        """
        logger.info(f"Task {task_id} completed by agent {agent_id} in {duration_ms}ms")
        
        # Update task status
        await update_task_status(
            task_id, 
            TaskStatus.COMPLETED, 
            result=result, 
            duration_ms=duration_ms
        )
        
        # Create task result record
        task_result = TaskResult(
            task_id=task_id,
            agent_id=agent_id,
            status="completed",
            result=result,
            duration_ms=duration_ms,
            model_used=model_used,
            tokens_used=tokens_used
        )
        
        await create_task_result(task_result)
        
        # Update trust metrics
        instances = await get_instances_with_trust_metrics()
        instance = next((inst for inst in instances if inst["instance"].id == agent_id), None)
        
        if instance:
            await increment_task_completed(instance["instance"].id, duration_ms)
        
        # Clear from performance cache
        # TODO: Implement cache invalidation
    
    async def handle_task_failure(
        self,
        task_id: UUID,
        error: str,
        agent_id: UUID,
        duration_ms: int,
        model_used: str
    ) -> None:
        """
        Handle task failure.
        
        Args:
            task_id: Failed task ID
            error: Error message
            agent_id: Agent that failed the task
            duration_ms: Time spent before failure
            model_used: LLM model used
        """
        logger.warning(f"Task {task_id} failed: {error}")
        
        # Update task status
        await update_task_status(
            task_id,
            TaskStatus.FAILED,
            result={"error": error}
        )
        
        # Create failure result record
        task_result = TaskResult(
            task_id=task_id,
            agent_id=agent_id,
            status="failed",
            result={"error": error},
            duration_ms=duration_ms,
            model_used=model_used,
            error_message=error
        )
        
        await create_task_result(task_result)
        
        # Update trust metrics
        instances = await get_instances_with_trust_metrics()
        instance = next((inst for inst in instances if inst["instance"].id == agent_id), None)
        
        if instance:
            await increment_task_failed(instance["instance"].id)
    
    async def get_routing_analytics(self) -> Dict[str, Any]:
        """
        Get routing analytics and performance metrics.
        
        Returns:
            Dict with analytics data for dashboard
        """
        # Get overall task metrics
        overall_metrics = await get_task_metrics()
        
        # Get per-agent metrics
        agent_metrics = await get_agent_task_metrics()
        
        # Calculate additional analytics
        analytics = {
            "overview": {
                "total_tasks": overall_metrics.total_tasks,
                "completed_tasks": overall_metrics.completed_tasks,
                "failed_tasks": overall_metrics.failed_tasks,
                "refused_tasks": overall_metrics.refused_tasks,
                "success_rate": overall_metrics.success_rate,
                "refusal_rate": overall_metrics.refusal_rate,
                "avg_completion_time_ms": overall_metrics.avg_completion_time_ms,
                "queue_depth": overall_metrics.queue_depth,
                "avg_queue_wait_time_ms": overall_metrics.avg_queue_wait_time_ms
            },
            "agent_performance": [
                {
                    "agent_id": str(metrics.agent_id),
                    "agent_role": metrics.agent_role,
                    "total_assigned": metrics.total_assigned,
                    "completed": metrics.completed,
                    "failed": metrics.failed,
                    "refused": metrics.refused,
                    "success_rate": metrics.success_rate,
                    "refusal_rate": metrics.refusal_rate,
                    "avg_duration_ms": metrics.avg_duration_ms,
                    "current_load": metrics.current_load
                }
                for metrics in agent_metrics
            ],
            "routing_efficiency": {
                # These would be calculated from more detailed metrics
                "first_choice_success_rate": 0.85,  # Placeholder
                "avg_routing_attempts": 1.2,  # Placeholder
                "override_rate": 0.05  # Placeholder
            }
        }
        
        return analytics


# Global task router instance
task_router = TaskRouter()