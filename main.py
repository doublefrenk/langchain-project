from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

load_dotenv()

model = os.getenv("LLM_MODEL")


def main():
    return ""


if __name__ == "__main__":
    main()
