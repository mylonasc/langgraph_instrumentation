from typing import Annotated, List, Literal, TypedDict, Any, Callable, Optional, Union, Awaitable
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
# from langchain_core.language_models import ChatOpenAI
from langchain.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define a type alias for the node functions (can be sync or async)
NodeCallable = Callable[[AgentState], Union[dict, Awaitable[dict]]]

def create_react_agent(
    model: BaseChatModel, 
    tools: List[BaseTool], 
    system_prompt: str = "",
    preprocess: Optional[NodeCallable] = None,
    postprocess: Optional[NodeCallable] = None,
):
    """Creates a compiled LangGraph ReAct agent with optional pre- and post-processing steps.

    This factory builds a state graph that cycles between an LLM ('agent') and tool execution 
    ('tools'). It allows for optional injection of logic before the agent starts (pre-processing) 
    and after the agent decides to stop (post-processing).

    Args:
        model (BaseChatModel): The LangChain chat model to use as the reasoning engine.
            Must support tool binding (.bind_tools).
        tools (List[BaseTool]): A list of tools the agent can access.
        system_prompt (str, optional): A system prompt to prepend to the message history 
            before every model call. Defaults to "".
        preprocess (Optional[NodeCallable], optional): A function (sync or async) to run 
            before the agent loop begins. It receives the current state and should return 
            a dictionary of state updates. Defaults to None.
        postprocess (Optional[NodeCallable], optional): A function (sync or async) to run 
            after the agent has decided to stop (i.e., when no tool calls are generated), 
            but before the graph ends. Defaults to None.

    Returns:
        CompiledStateGraph: A compiled LangGraph application that can be invoked or 
        streamed.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage

        # 1. Define custom nodes
        async def sanitize_input(state):
            # Add a security warning before the agent starts
            return {"messages": [SystemMessage(content="SECURITY MODE: ACTIVE")]}

        async def format_output(state):
            # Log the final response length
            last_msg = state["messages"][-1]
            print(f"Final response length: {len(last_msg.content)}")
            return {}

        # 2. Create the agent
        model = ChatOpenAI(model="gpt-4")
        tools = [] # Add your tools here

        app = create_react_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant.",
            preprocess=sanitize_input,
            postprocess=format_output
        )

        # 3. Run the agent
        final_state = await app.ainvoke({"messages": [("user", "Hello")]})
        ```
    """
    # 1. Bind tools
    model_with_tools = model.bind_tools(tools)

    # 2. Define Async Agent Node
    async def call_model(state: AgentState):
        messages = state["messages"]
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages
        
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # 3. Conditional Logic
    def should_continue(state: AgentState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # 4. Build Graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    if preprocess:
        workflow.add_node("preprocess", preprocess)
    if postprocess:
        workflow.add_node("postprocess", postprocess)

    # 5. Define Edges
    if preprocess:
        workflow.add_edge(START, "preprocess")
        workflow.add_edge("preprocess", "call_model")
    else:
        workflow.add_edge(START, "call_model")

    end_destination = "postprocess" if postprocess else END
    
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            END: end_destination
        }
    )

    workflow.add_edge("tools", "call_model")

    if postprocess:
        workflow.add_edge("postprocess", END)

    return workflow.compile()