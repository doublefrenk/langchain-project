from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import Tavily
import os

load_dotenv()

model = os.getenv("LLM_MODEL")


def main():
    print("Welcome to the Groq LLM demo!")

if __name__ == "__main__":
    main()
