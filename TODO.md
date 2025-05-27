# TODO.md

## UI
- Need to update padding on .mdc-card-header to have a padding-bottom of 16px to match
    - alternatively could do this on the .mdc-card-content for padding-top

- Need to update notification left hand panel to ensure that statuses are taken into consideration for the display of what icon is available
    - Completed - green/blueish - current - ping circle
    - Awaiting Human-in-the-loop - darker orange - exclamation warning for immediate action as the agents are on hold until HITL action takes place
    - Pending - either awaiting response from other agent, system call, API call, MCP call etc. - some sort of waiting icon - yellowish
    - Error - error at any point along the trajectory of the workflow or the agent tool call or actions (anything within the CORE system) - danger image, red obviously



## Backend
CORE loop

Comprehension:
- Take in user (or agent) input, tool call, response, etc., and run a "Comprehension Check"
    - Check semantic similarity against vector DB of global knowledgebase
    - Check semantic similarity against Capabilities Check
        - System Capabilities first:
            - Services & APIs description of what is available to use for the CORE system engine
            - 

