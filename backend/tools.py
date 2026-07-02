import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)

# Il metodo `retrieve_context` è un tool che permette di recuperare il contesto rilevante per una query specifica. Utilizza il vector store per cercare documenti simili alla query e restituisce sia il contenuto dei documenti e eventuali artefatti associati.
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
  """Retrieve relevant documentation to help answer user queries about Langchain."""

  retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

  # Serialize documents for the model
  serialized = "\n\n".join(
      (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
      for doc in retrieved_docs
  )

  return serialized, retrieved_docs
