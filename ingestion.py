import asyncio
import os
import ssl
from typing import List, Dict, Any
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_info, log_success, log_error, log_warning, log_header


load_dotenv()

# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds= 10)
# chroma = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
tavily_extract = TavilyExtract()

# Questo è il modulo che ci permette di fare scraping e crawling dei siti web, è molto potente e ci permette di estrarre informazioni in modo strutturato, è fondamentale per il nostro processo di ingestione dei dati. Lo useremo per estrarre i contenuti dai siti web che vogliamo indicizzare e poi li passeremo al nostro modello di embedding per creare rappresentazioni vettoriali che potremo poi utilizzare per le nostre ricerche semantiche.
tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap(map_depth=5, max_breadth=20, max_pages=1000)

def chunks_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(
            f"🔄 TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs.",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(
            input={"urls": urls, "extract_depth": "advanced"}
        )
        extracted_docs_count = len(docs.get("results", []))
        if extracted_docs_count > 0:
            log_success(
                f"TavilyExtract: Completed batch {batch_num} - extracted {extracted_docs_count} documents"
            )
        else:
            log_error(
                f"TavilyExtract: Batch {batch_num} failed to extract any documents, {docs}"
            )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"🔧 TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )
    # Questo è un passaggio cruciale: stiamo raccogliendo delle coroutine (o delle Promise se preferisci) che poi verranno consumate solo in seguito. Questo perchè extract_batch è una funzione asincrona che restituisce una coroutine, quindi quando la chiamiamo non stiamo eseguendo l'estrazione immediatamente, ma stiamo creando una lista di coroutine che rappresentano le estrazioni da eseguire. In questo modo possiamo poi utilizzare asyncio.gather per eseguire tutte queste estrazioni in parallelo, il che è molto più efficiente rispetto a eseguirle una alla volta.
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

    # Pensalo come un Promise.all in JavaScript, stiamo lanciando tutte le estrazioni in parallelo e poi aspettiamo che tutte siano completate. Questo ci permette di velocizzare notevolmente il processo di estrazione, soprattutto quando abbiamo un gran numero di URL da processare. Inoltre, gestiamo eventuali errori a livello di batch, in modo da poter continuare con le altre estrazioni anche se una batch fallisce, e alla fine filtriamo i risultati per ottenere solo quelli validi.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:  # type: ignore
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages

async def main():
  """Main async function to orchestrate the ingestion process."""

  log_header("Starting Ingestion Process")

  log_info("Tavily Map: Starting to map the langchain documentation from https://python.langchain.com/", Colors.PURPLE)

  #  Crawl documentation site

  # Il campo instructions è molto importante, è quello che dice a tavily cosa deve estrarre dalle pagine web, in questo caso vogliamo estrarre solo il contenuto relativo agli agenti AI, quindi è fondamentale essere specifici nelle istruzioni per ottenere risultati di qualità. Inoltre, max_depth e extract_depth ci permettono di controllare quanto in profondità vogliamo che tavily esplori il sito e quanto dettagliatamente vogliamo che estragga le informazioni, è un bilanciamento tra quantità e qualità dei dati estratti.
  # res = tavily_crawl.invoke({
  #   "url": "https://python.langchain.com/",
  #   "max_depth": 5,
  #   "extract_depth": "advanced",
  # })

  # all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]

  # log_success(f"Crawled {len(all_docs)} documents from the documentation site.")

  # Mapping documentation site with Tavily Map

  site_map = tavily_map.invoke("https://python.langchain.com/")

  log_success(f"Mapped the site with {len(site_map['results'])} nodes.")

  urls_batches = chunks_urls(list(site_map["results"]), chunk_size=20)

  all_docs = await async_extract(urls_batches)


if __name__ == "__main__":
    asyncio.run(main())
