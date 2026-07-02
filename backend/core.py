from typing import Any, Dict
from constants import SYSTEM_PROMPT

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from tools import retrieve_context


# Initialize chat model - questo è un modo del tutto nuovo di inizializzare un modello di chat, in questo caso stiamo usando il modello "gpt-5.2" fornito da OpenAI. Questo modello sarà utilizzato per generare risposte alle query degli utenti basate sui documenti indicizzati nel nostro vector store.
model = init_chat_model("gpt-5.2", model_provider="openai")

def run_llm(query:str) -> Dict[str, Any]:
  """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

  system_prompt = SYSTEM_PROMPT

  agent = create_agent(
      model=model,
      tools=[retrieve_context],
      system_prompt=system_prompt
  )

  messages = [{"role": "user", "content": query}]

  # Run the agent with the tool message. L'agent vuole sempre il messaggio nel formato {"messages": messages}, quindi dobbiamo passare i messaggi in questo formato. In questa fase l'agente cerca di capire se ha bisogno di usare uno strumento (in questo caso il tool `retrieve_context`) per recuperare informazioni rilevanti dai documenti indicizzati. Se l'agente decide di usare lo strumento, genererà un messaggio di tipo ToolMessage che conterrà la query da passare al tool.
  response = agent.invoke({"messages": messages})

# Recupero il contenuto della risposta generata dall'agent. L'ultima voce nella lista dei messaggi contiene la risposta finale dell'agent, quindi la estraiamo da lì.
  answer = response["messages"][-1].content

  # Extract context documents from ToolMessage artifacts
  context_docs = []
  for message in response["messages"]:
      # Check if this is a ToolMessage with artifact
      if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
          # The artifact should contain the list of Document objects
          if isinstance(message.artifact, list):
              context_docs.extend(message.artifact)

  return {
      "answer": answer,
      "context": context_docs
  }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)

