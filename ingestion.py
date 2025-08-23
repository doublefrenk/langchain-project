import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader('./mediumblog1.txt')

    # Questo metodo carica il contenuto del file 'mediumblog1.txt' e lo memorizza nella variabile 'document'di Langchain
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv('PINECONE_INDEX_NAME'))
    print("Done!")

