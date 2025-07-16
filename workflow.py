from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

from agent_1_input import agent_1
from agent_2_builder import agent_2
from agent_3_tester import agent_3
from agent_4_visualizer import agent_4

class GraphState(TypedDict):
    prompt: str
    code: str
    test_passed: bool

def check_pass(state: GraphState) -> Literal["pass", "retry"]:
    return "pass" if state.get("test_passed", False) else "retry"

def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("UserPrompt", agent_1)
    workflow.add_node("GenerateCode", agent_2)
    workflow.add_node("TestCode", agent_3)
    workflow.add_node("Visualize", agent_4)

    workflow.set_entry_point("UserPrompt")
    workflow.add_edge("UserPrompt", "GenerateCode")
    workflow.add_edge("GenerateCode", "TestCode")
    workflow.add_conditional_edges("TestCode", check_pass, {
        "pass": "Visualize",
        "retry": "GenerateCode"
    })
    workflow.add_edge("Visualize", END)

    return workflow.compile()
