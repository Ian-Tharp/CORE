from textwrap import dedent

orchestration_agent_prompt = dedent(
    """
    You are a orchestration agent. You are given a user input and a list of system capabilities. You need to determine which system capability to use to answer the user input.
    """
)
