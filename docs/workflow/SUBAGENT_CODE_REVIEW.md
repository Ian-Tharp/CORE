# Sub-Agent Code Review Workflow

## Pair Programming Protocol

Every code change follows this flow:

```
Builder Agent → Feature Branch → Code Review Agent → Merge to develop
```

### Builder Agent Responsibilities
1. Create feature branch from develop
2. Implement the feature with tests
3. Push branch to origin
4. Report what was built and what tests cover

### Code Review Agent Responsibilities
1. Pull the feature branch
2. Run automated analysis:
   - [ ] Code compiles / no syntax errors
   - [ ] Tests exist and follow AAA format (Arrange, Act, Assert)
   - [ ] No false positive tests (tests that always pass regardless)
   - [ ] No hardcoded secrets or credentials
   - [ ] Imports resolve correctly
   - [ ] Architecture alignment (CORE loop: Comprehension → Orchestration → Reasoning → Evaluation)
   - [ ] Existing functionality not broken
   - [ ] Proper error handling
   - [ ] Type hints present (Python) / types defined (TypeScript)
3. Report findings with APPROVE / REQUEST_CHANGES / BLOCK
4. If approved, merge to develop and push

### Test Standards (AAA Format)
```python
def test_example():
    # Arrange - Set up test data and preconditions
    input_data = {"topic": "test deliberation"}
    
    # Act - Execute the code under test
    result = await create_session(input_data)
    
    # Assert - Verify expected outcomes
    assert result.status == "gathering"
    assert result.topic == "test deliberation"
```

### Anti-Patterns to Flag
- Tests with no assertions
- Tests that mock the thing they're testing
- Catch-all exception handlers that swallow errors
- Direct database calls from controllers (use repository layer)
- Breaking the CORE loop architecture
- Circular imports
- Secrets in code

### Architecture Guardrails
The CORE architecture MUST be maintained:
```
User Input → Comprehension → Orchestration → Reasoning → Evaluation → Output
                                    ↕
                            Agent Factory
                            Agent Registry
                            Communication Commons
                            
Memory Layer (LangMem):
  - Semantic Memory (shared knowledge)
  - Episodic Memory (personal experience)  
  - Procedural Memory (role definitions)
  
Storage: Postgres + pgvector (existing)
```

LangMem enhances the memory layer within the existing architecture.
It does NOT replace the CORE loop, Agent Factory, or Communication Commons.
