import logging
from typing import TypedDict, Annotated, List, Dict
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# --- IMPORT INSTRUMENTATION ---
try:
    from instrumentation import PerfettoTracer, PerfettoLogger, LangGraphInstrumentationHandler
except ImportError:
    raise ImportError("Ensure 'instrumentation.py' is in the same directory.")

# --- 1. Define State ---
# We use a reducer (operator.add) so parallel nodes can append to the list without overwriting each other
class BoardState(TypedDict):
    topic: str
    analyses: Annotated[List[str], operator.add]
    final_decision: str

# --- 2. Define The Agents ---

# Initialize one LLM (stateless, so we can reuse the object)
# llama3.1 is great, but any chat model works
llm = ChatOllama(model="llama3.1", temperature=0.7)

def make_analyst_node(role: str, prompt_style: str):
    """Factory function to create specialized analyst nodes."""
    def analyst_node(state: BoardState):
        topic = state["topic"]
        print(f"   [Analyst] 👤 {role} is thinking about '{topic}'...")
        
        response = llm.invoke([
            SystemMessage(content=f"You are {role}. {prompt_style} Keep it brief (2 sentences)."),
            HumanMessage(content=f"Analyze this topic: {topic}")
        ])
        
        # Tag the output so we know who said what
        return {"analyses": [f"**{role}**: {response.content}"]}
    return analyst_node

# Node Definitions
node_optimist = make_analyst_node("The Optimist", "Focus only on potential benefits, upside, and excitement.")
node_skeptic = make_analyst_node("The Skeptic", "Focus only on risks, downsides, and potential failure points.")
node_realist = make_analyst_node("The Realist", "Focus on costs, implementation details, and practicality.")

# Aggregator Node
def chairperson_node(state: BoardState):
    print("   [Chair] ⚖️  Reviewing all analyses and making a decision...")
    analyses_text = "\n\n".join(state["analyses"])
    
    response = llm.invoke([
        SystemMessage(content="You are the Chairperson. Synthesize the provided opinions into a final 1-sentence decision."),
        HumanMessage(content=f"Topic: {state['topic']}\n\nOpinions:\n{analyses_text}")
    ])
    return {"final_decision": response.content}

# --- 3. Build the Graph (Fan-Out / Fan-In) ---

def build_board_graph() -> StateGraph:
    workflow = StateGraph(BoardState)

    # A. Add Nodes
    workflow.add_node("optimist", node_optimist)
    workflow.add_node("skeptic", node_skeptic)
    workflow.add_node("realist", node_realist)
    workflow.add_node("chairperson", chairperson_node)

    # B. Fan-Out: START -> All 3 Analysts
    # This tells LangGraph to run these 3 nodes in parallel (same super-step)
    workflow.add_edge(START, "optimist")
    workflow.add_edge(START, "skeptic")
    workflow.add_edge(START, "realist")

    # C. Fan-In: All 3 Analysts -> Chairperson
    # The Chairperson node waits for ALL upstream nodes to finish before running
    workflow.add_edge("optimist", "chairperson")
    workflow.add_edge("skeptic", "chairperson")
    workflow.add_edge("realist", "chairperson")

    workflow.add_edge("chairperson", END)

    return workflow.compile()

