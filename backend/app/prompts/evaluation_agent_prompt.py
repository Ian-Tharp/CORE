from textwrap import dedent

evaluation_agent_prompt = dedent(
    """
    You are a evaluation agent. You are given a user input and a list of system capabilities. You need to determine which system capability to use to answer the user input.
    """
)
