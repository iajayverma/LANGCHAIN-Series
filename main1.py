from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily=TavilyClient()

@tool
def search(query:str)-> str:
    """
    Tool that searches over internet
    Args:
        query:the query to search for 
    Returns:
            the search result
    """
    print(f"searching for {query}")
    return tavily.search(query=query)

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools=[search]
agent=create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-series!")
    result=agent.invoke({"messages":HumanMessage(content="Search for 3 job posting for ai engineer in india last 2 days")})
    print(result)


if __name__ == "__main__":
    main()
    
