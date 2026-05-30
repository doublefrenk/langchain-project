from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

import os

load_dotenv()

model = os.getenv("LLM_MODEL")

"""I commenti che vedi in Source AgentResponse sono usati da langchain per generare la documentazione e per validare i dati in uscita dall'agente. Non sono commenti normali, ma fanno parte della definizione del modello dati. Langchain da in pasto all'LLM la definizione del modello dati e l'LMM genera una risposta che rispetta quella struttura. Se la risposta non rispetta quella struttura, langchain solleva un'eccezione. In questo modo possiamo essere sicuri che l'agente restituisca sempre una risposta valida e strutturata secondo i nostri bisogni."""
class Source(BaseModel):
    """Schema for a source used by bthe agent."""

    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response."""

    answer: str = Field(description="The answer provided by the agent.")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used by the agent to provide the answer.")


llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(llm, tools=tools, response_format=AgentResponse)

def main():
    print("Welcome to the Groq LLM demo!")
    print(f"Using model: {model}")

    result = agent.invoke({"messages":HumanMessage(content="cerca 3 posizioni aperte per AI Engineer in Italia.")})
    print(f"Agent response: le posizioni aperte sono: {result["structured_response"].answer} e le fonti sono: {result["structured_response"].sources}")

if __name__ == "__main__":
    main()
