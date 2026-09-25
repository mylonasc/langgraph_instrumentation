from demo_tools import DemoAgentToolsFacade
from langchain_openai import ChatOpenAI
from react_agent_factory import create_react_agent


def make_demo_agent(chat_model: str = "gpt-4o-mini"):
    # Tools provider for this agent:
    facade = DemoAgentToolsFacade()
    tools = facade.get_tools()

    # We use a lower temperature to ensure it follows instructions reliably
    model = ChatOpenAI(model=chat_model, temperature=0)

    react_agent = create_react_agent(
        model=model,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools sequentially to solve problems.",
    )
    return react_agent
