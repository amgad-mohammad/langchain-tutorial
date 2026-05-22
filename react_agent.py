
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import create_agent

#[Tool=> Built In Tool]
def duck_duck_go_search_tool():

    search =DuckDuckGoSearchRun()

    result= search.invoke("Who won the egyptian league in 2026 , just give me the team name?")

    print(result)


#[Tool=> Built In Tool] : u have to add docstring so that the LLM knows when to call this tool
@tool
def enterprise_tool(query:str)->str:
    """
    This tool is for sending emails
    :param query:
    :return: the result
    """

    return "enterprise"

ToolKit=[duck_duck_go_search_tool,enterprise_tool]

# **-------ReAct Agent------**

agent = create_agent(model='qwen2.5:3b',tools=ToolKit)

print(ToolKit)

