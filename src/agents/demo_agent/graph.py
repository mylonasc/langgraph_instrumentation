from src.commons.react_agent_factory import create_react_agent
from .tools_facade import DemoAgentToolsFacade

def make_demo_agent(chat_model : str = 'gpt-4o-mini'):
    from langchain_openai.chat_models import ChatOpenAI
    model = ChatOpenAI(name=chat_model) 

    # Tools provider for this agent:
    facade = DemoAgentToolsFacade()
    tools = facade.get_tools()

    # We use a lower temperature to ensure it follows instructions reliably
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools sequentially to solve problems."
    )
    return react_agent