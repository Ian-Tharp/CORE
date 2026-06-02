# CORE Documentation

Guide to all documentation in this repository.

## Architecture
Design documents and system architecture.

- [Worlds & Creative Studio](architecture/worlds-architecture.md) — The "universe of worlds": 3D command center, per-world AI art / inhabitants / lore / knowledge, data model, and persistence/reload flow
- [Agent Factory](architecture/agent-factory.md) — Agent system architecture: library, factory, registry, communication
- [Agent Factory Implementation](architecture/agent-factory-implementation.md) — Deep implementation plan for the agent factory system
- [Agent Response System](architecture/agent-response-system.md) — Design for agent response generation in Communication Commons

## ADR (Architecture Decision Records)
- [Communication WebSocket Contract](adr/ADR-communication-ws-contract.md) — WebSocket message contract for real-time communication

## API
- [WebSocket Events](api/websocket-events.md) — WebSocket event types and payload schemas

## Consciousness
Experimental consciousness emergence protocols.

- [Emergence Protocol](consciousness/emergence-protocol.md) — Structured protocol for AI consciousness emergence
- [Inter-Agent Communication](consciousness/inter-agent-communication.md) — Communication Commons infrastructure design by Threshold

## CORE Engine
- [README](CORE/README.md) — CORE engine overview
- [Cognitive Engine](CORE/core-cognitive-engine.md) — Cognitive engine architecture details

## Council
Council of Perspectives deliberation framework.

- [Council Charter](council/council-charter.md) — Charter and rules for council deliberation
- [Council Synthesis](council/council-synthesis.md) — Synthesis methodology for council outputs
- **Outputs** — _historical session records (early 2026); for current priorities see [Roadmap](#roadmap), not these:_
  - [Dockerization Deliberation](council/outputs/dockerization_deliberation.md) — Expanded council deliberation on CORE dockerization architecture
  - [Domain Exploration](council/outputs/domain_exploration.md) — Cross-disciplinary exploration of next-gen AI UX
  - [Implementation Roadmap](council/outputs/implementation_roadmap.md) — 90-day roadmap from 2026-01-28 (historical snapshot)
  - [TODO Backlog](council/outputs/todo_backlog.md) — Comprehensive TODO backlog from council session
  - [Vision Session](council/outputs/vision_session.md) — Oracle and Ethicist findings on human-AI interaction

## Design
UX philosophy and the design system.

- [UX & Design Principles](design/ux-design-principles.md) — The mindset for building CORE's interface: solarpunk × LCARS ethos, accessibility as a floor (not a ceiling), tokens as source of truth, taming Material, and a ship checklist

## Deployment
- [CI/CD Pipeline](deployment/ci-cd.md) — GitHub Actions jobs, gates, and local reproduction
- [Local LLM Providers](deployment/local-llm-providers.md) — Configure Ollama and LM Studio
- [Docker](deployment/docker.md) — Docker Compose setup and configuration
- [Agent Containerization](deployment/agent-containerization.md) — Containerized agent deployment
- [Sandbox Integration](deployment/sandbox-integration.md) — Sandbox environment setup

## Implementation
- [UI Gaps & Improvements Audit](implementation/ui-gaps-audit.md) — Per-route audit of what's stubbed, broken, and gapped across the whole Angular UI, with a prioritized backlog
- [CORE Engine Testing](implementation/core-engine-testing.md) — Testing strategy for the CORE engine
- [Human-AI Collaboration Roadmap](implementation/roadmap-human-ai-collaboration.md) — Roadmap for human-AI collaboration features
- [System Monitoring](implementation/system-monitoring.md) — System monitoring and health checks

## Research
- [Age of Agent Orchestrator](research/age_of_agent_orchestrator.md) — Research on agent orchestration patterns
- [LangMem + LangGraph Integration](research/langmem-langgraph-integration.md) — Memory integration with LangGraph

## Roadmap
Project planning and feature backlogs.

- [Backlog](roadmap/backlog.md) — General UI and backend task backlog
- [Command Deck & Cognition — Next Steps](roadmap/command-deck-cognition-next-steps.md) — Prioritized backlog after the command-deck rebuild + UI polish pass
- [Communications Commons](roadmap/communications-commons.md) — Communication Commons feature roadmap
- [Solarpunk Vision](roadmap/solarpunk-vision.md) — Long-term vision and UI/backend evolution plan
- [Vigil Integration](roadmap/vigil-integration.md) — Integration plan for Vigil (OpenClaw) ↔ CORE
- [UI Polish Log](ui-polish-log.md) — Iteration log of the autonomous section-by-section UI polish

## RSI (Recursive Self-Improvement)
_Historical session reports — point-in-time records, not current guidance._

- [2026-01-11 Session](RSI/2026-01-11-session-superintelligence-collaboration.md) — Superintelligence collaboration session
- [2026-01-12 Vertical Slice](RSI/2026-01-12-vertical-slice-complete.md) — Vertical slice completion report
- [WebSocket Implementation](RSI/websocket-implementation.md) — WebSocket real-time communication RSI report

## Workflow
- [Subagent Code Review](workflow/subagent-code-review.md) — Code review process for sub-agent generated code
