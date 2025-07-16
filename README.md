# AI
The aim of this LangGraph is to establish a communication between AI Agents allowing them to call each other and it's functions required to successfully simulate a website as requested by the user

Working:
>Agent 1 accepts User input as a prompt. Then the Agent 2 builds code, which is organised systematically by the Agent 3, and the Agent 4 visualises it, returns any errors or exceptions if found to  the Agent 2

Communication path followed:
>Agent 1 and Agent 2 and Agent 3 communicate in a linear sequence, while Agent 2 and Agent 4 communicate in a back-and-forth loop until approved, and the result is then sent to Agent 4

Key Points to remember:
> This is just a framework connecting seperate AI Agents.
> The successful functioning of the Project depends on the individual functioning of the Agents
> The LangGraph serves the purpose of giving a framework platform to allow the agents to access each other 
