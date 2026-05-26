from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
import os

load_dotenv()

model = os.getenv("LLM_MODEL")



llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(llm, tools=tools)

def main():
    print("Welcome to the Groq LLM demo!")
    print(f"Using model: {model}")

    result = agent.invoke({"messages":HumanMessage(content="cerca 3 posizioni aperte per AI Engineer in Italia.")})
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()
