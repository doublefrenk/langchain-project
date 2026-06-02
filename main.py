import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Initializing...")

embedding = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(embedding=embedding, index_name=os.getenv('PINECONE_INDEX_NAME'))

# In questo caso trattiamo il vector store come un retriever, ovvero un componente che ci permette di recuperare i documenti più rilevanti in base a una query. In questo caso, stiamo specificando che vogliamo recuperare i 3 documenti più rilevanti (k=3) per ogni query che faremo al retriever.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
  """Answer the question based only on the following context:
  {context}

  Question: {question}

  Answer:"""
)

def create_retriever_chain():
    """
    Create a retreival chain using Langchain Expression Language).
    Returns a chian that can be invoked  with {question} as input.
    """
    # In teoria questa chain ha dei problemi: format_doc è una regolare funzione python e quindi non ha il metodo invoke() come gli altri oggetti di langchain, ma il solo fatto di essere in una pipe con altri oggetti langchain fa capire a python che deve essere trattato come uno di questi
    # L'altro problema che si incontra è che prompt_template vuole due argomenti, tra cui il context è anche dinamico. la strategia è usare RunnablePassthrough che agisce come un nastro trasportatore per oggetti, ovvero l'input non viene modificato. In questo caso
    # gli stiamo esplicitamente dicendo di costruire un altro campo dell'oggetto che è proprio il contesto contesto.
    retriever_chain = RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_doc ) | prompt_template | llm | StrOutputParser()
    return retriever_chain

# Questa funzione ci permette di formattare i documenti recuperati del retriever in un formato che può essere inserito come contesto
def format_doc(docs):
    return "/n/n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    print("Retrieving...")

    question = "What is Pinecone in machine learning?"

    chain = create_retriever_chain()
    answer = chain.invoke({"question":question})

    print(f"Question: {question}\n\n Answer: {answer}")